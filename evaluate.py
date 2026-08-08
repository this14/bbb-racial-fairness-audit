"""
evaluate.py

Evaluates the trained model ONCE on the fixed, subject-level held-out test set
(from train.py's split_indices.npz), then reports AUROC / sensitivity / specificity
overall and per race/ethnicity subgroup, with:
  - DeLong's test for pairwise AUROC differences between subgroups
  - Bootstrap (n=1000) 95% CIs for all point estimates

Reads from the same X.npy/y.npy/groups.npy/subjects.npy folder produced by
preprocess.py (see train.py's header comment for why plain .npy files, not a
single .npz, are used -- reliable mmap).

Usage:
    python evaluate.py --data_dir ./preprocessed_27178 \
        --model_dir ./model_out_27178 --out_dir ./eval_out_27178
"""
import argparse
import itertools
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from resnet1d import ResNet1D34


class MemmapECGDataset(Dataset):
    def __init__(self, x_npy_path, row_indices):
        self.x_npy_path = x_npy_path
        self.row_indices = np.asarray(row_indices)
        self._X = None

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.x_npy_path, mmap_mode='r')

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        self._ensure_open()
        row = self.row_indices[idx]
        x = np.array(self._X[row])
        return torch.from_numpy(x).float(), idx


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42, threshold=None):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        if threshold is None:
            stats.append(metric_fn(yt, ys))
        else:
            stats.append(metric_fn(yt, ys, threshold))
    stats = np.array(stats)
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def sensitivity(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else float('nan')


def specificity(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    tn = ((pred == 0) & (y_true == 0)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else float('nan')


def delong_variance(y_true, y_score):
    """DeLong variance of a single AUROC estimate (Sun & Xu, 2014 fast DeLong)."""
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    m = int(y_sorted.sum())
    n = len(y_sorted) - m
    positive = scores_sorted[:m]
    negative = scores_sorted[m:]

    tx = compute_midrank(positive)
    ty = compute_midrank(negative)
    tz = compute_midrank(scores_sorted)

    auc = tz[:m].sum() / m / n - (m + 1) / (2 * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return auc, var


def independent_auc_test(y1, s1, y2, s2):
    """
    Two-sided z-test for AUROC(group1) != AUROC(group2), where group1 and group2 are
    INDEPENDENT samples (different patients) -- NOT the paired/same-test-set case that
    DeLong's original test was designed for. Variance of each AUROC is estimated via
    DeLong's method and combined assuming independence: Var(auc1 - auc2) = Var(auc1) + Var(auc2).
    """
    auc1, var1 = delong_variance(y1.astype(float), s1)
    auc2, var2 = delong_variance(y2.astype(float), s2)
    se = np.sqrt(var1 + var2)
    z = (auc1 - auc2) / se if se > 0 else np.nan
    from scipy import stats as sstats
    p = 2 * (1 - sstats.norm.cdf(abs(z)))
    return auc1, auc2, p


def run_inference(x_npy_path, row_indices, model, device, batch_size=64, num_workers=0):
    ds = MemmapECGDataset(x_npy_path, row_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = np.empty(len(ds), dtype=np.float32)
    with torch.no_grad():
        for xb, idxs in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs[idxs.numpy()] = p
    return probs


def main(data_dir, model_dir, out_dir, batch_size=64, num_workers=0):
    os.makedirs(out_dir, exist_ok=True)

    x_npy_path = os.path.join(data_dir, 'X.npy')
    y = np.load(os.path.join(data_dir, 'y.npy'), allow_pickle=True)
    groups = np.load(os.path.join(data_dir, 'groups.npy'), allow_pickle=True)
    subjects = np.load(os.path.join(data_dir, 'subjects.npy'), allow_pickle=True)

    split = np.load(os.path.join(model_dir, 'split_indices.npz'), allow_pickle=True)
    test_subj = set(split['test_subj'])
    test_mask = np.isin(subjects, list(test_subj))
    test_rows = np.where(test_mask)[0]
    y_test, groups_test = y[test_mask], groups[test_mask]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D34().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location=device))
    model.eval()

    probs = run_inference(x_npy_path, test_rows, model, device, batch_size=batch_size, num_workers=num_workers)

    overall_auc = roc_auc_score(y_test, probs)
    lo, hi = bootstrap_ci(y_test, probs, roc_auc_score)
    print(f"Overall test AUROC: {overall_auc:.3f} [95% CI: {lo:.3f}-{hi:.3f}]  N={len(y_test)}")

    subgroup_results = {}
    for g in sorted(set(groups_test)):
        mask = groups_test == g
        yg, pg = y_test[mask], probs[mask]
        if len(np.unique(yg)) < 2:
            print(f"  {g}: SKIPPED (only one class present in test set, n={mask.sum()})")
            continue
        auc_g = roc_auc_score(yg, pg)
        auc_lo, auc_hi = bootstrap_ci(yg, pg, roc_auc_score)
        sens_g = sensitivity(yg, pg)
        sens_lo, sens_hi = bootstrap_ci(yg, pg, sensitivity, threshold=0.5)
        spec_g = specificity(yg, pg)
        spec_lo, spec_hi = bootstrap_ci(yg, pg, specificity, threshold=0.5)
        subgroup_results[g] = dict(auc=auc_g, auc_ci=(auc_lo, auc_hi),
                                    sens=sens_g, sens_ci=(sens_lo, sens_hi),
                                    spec=spec_g, spec_ci=(spec_lo, spec_hi), n=mask.sum())
        print(f"  {g}: AUROC={auc_g:.3f} [{auc_lo:.3f}-{auc_hi:.3f}], "
              f"Sens={sens_g:.3f} [{sens_lo:.3f}-{sens_hi:.3f}], "
              f"Spec={spec_g:.3f} [{spec_lo:.3f}-{spec_hi:.3f}]  (n={mask.sum()})")

    print("\nPairwise test for AUROC differences between subgroups")
    print("(independent-samples z-test using DeLong variance per group; subgroups are")
    print(" different patients, so this is NOT the paired same-test-set DeLong comparison):")
    valid_groups = list(subgroup_results.keys())
    for g1, g2 in itertools.combinations(valid_groups, 2):
        m1, m2 = groups_test == g1, groups_test == g2
        y1, s1 = y_test[m1].astype(float), probs[m1]
        y2, s2 = y_test[m2].astype(float), probs[m2]
        if len(np.unique(y1)) < 2 or len(np.unique(y2)) < 2:
            print(f"  {g1} vs {g2}: SKIPPED (one class missing in a subgroup)")
            continue
        auc1, auc2, p = independent_auc_test(y1, s1, y2, s2)
        print(f"  {g1} (AUROC={auc1:.3f}) vs {g2} (AUROC={auc2:.3f}): p={p:.4f}")

    import json
    with open(os.path.join(out_dir, 'subgroup_results.json'), 'w') as f:
        json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else float(vv))
                        for kk, vv in v.items()} for k, v in subgroup_results.items()}, f, indent=2)
    print(f"\nSaved results to {out_dir}/subgroup_results.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                     help='Folder containing X.npy, y.npy, groups.npy, subjects.npy (from preprocess.py)')
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out_dir', default='./eval_out')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()
    main(args.data_dir, args.model_dir, args.out_dir, args.batch_size, args.num_workers)
