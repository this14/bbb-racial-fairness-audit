import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from train_naft import NAFTMemmapDataset, NAFTResNet1D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--split_dir", type=str, default="./imbalanced_split")
    parser.add_argument("--model_dir", type=str, default="./model_naft_balanced")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    groups_path = os.path.join(args.data_dir, "groups.npy")
    test_idx = np.load(os.path.join(args.split_dir, "test_idx.npy"))

    test_ds = NAFTMemmapDataset(x_path, y_path, groups_path, test_idx)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = NAFTResNet1D(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model_naft_balanced.pt"), map_location=device))
    model.eval()

    all_probs, all_targets, all_groups = [], [], []

    with torch.no_grad():
        for x_clean, yb, gb in test_loader:
            x_clean = x_clean.to(device)
            noise = torch.randn_like(x_clean) * 0.15
            x_noisy = x_clean + noise

            logits, _, _ = model(x_noisy)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(yb.numpy())
            all_groups.extend(gb.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_groups = np.array(all_groups)

    # 1. Optimal Threshold 탐색 (Max F1-score 기준)
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    opt_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.15

    print(f"\n[INFO] Probability Range: {all_probs.min():.4f} ~ {all_probs.max():.4f}")
    print(f"[INFO] Optimal Threshold: {opt_threshold:.4f} (Max F1: {f1_scores[opt_idx]:.4f})")

    # 2. 서브그룹별 성능 재산출
    group_names = {0: 'Asian', 1: 'Black', 2: 'Hispanic/Latino', 3: 'White', 4: 'Other'}
    results = []
    tpr_list, fpr_list = [], []

    for g_id, g_name in group_names.items():
        mask = (all_groups == g_id)
        if np.sum(mask) == 0:
            continue
            
        sub_probs = all_probs[mask]
        sub_targets = all_targets[mask]
        
        auroc = roc_auc_score(sub_targets, sub_probs)
        pred_bin = (sub_probs >= opt_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(sub_targets, pred_bin, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        results.append({
            'Subgroup': g_name,
            'N': np.sum(mask),
            'AUROC': round(auroc, 4),
            'Sensitivity (TPR)': round(tpr, 4),
            'Specificity (1-FPR)': round(1.0 - fpr, 4),
            'FPR': round(fpr, 4)
        })

    eq_odds_gap = (max(tpr_list) - min(tpr_list) + max(fpr_list) - min(fpr_list)) / 2.0
    df_res = pd.DataFrame(results)

    print("\n=========================================================================")
    print(f"  Balanced NAFT Evaluation (Optimal Threshold = {opt_threshold:.4f})     ")
    print("=========================================================================")
    print(df_res.to_string(index=False))
    print("-------------------------------------------------------------------------")
    print(f" Equalized Odds Gap (Fairness Metric): {eq_odds_gap:.4f}")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()