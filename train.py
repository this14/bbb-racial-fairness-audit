"""
train.py

Trains the 1D ResNet-34 BBB detector using a SUBJECT-LEVEL 70/10/20 train/val/test split
(no patient appears in more than one split).

MEMORY NOTE (important -- read before changing --data_dir):
Earlier versions loaded a single preprocessed_dataset.npz and relied on
np.load(..., mmap_mode='r') to avoid pulling the full ~6GB X array into RAM.
That did NOT work reliably: numpy's mmap support for arrays stored inside a
zip container (.npz) is unreliable across versions/platforms, and on this
machine it silently fell back to a full eager read, which crashed with
MemoryError.

preprocess.py now saves X, y, groups, subjects as SEPARATE plain .npy files
(no zip container). A plain .npy file supports real, reliable memory-mapping,
so mmap_mode='r' on X.npy actually keeps the array on disk, and only the rows
we index in MemmapECGDataset.__getitem__ get read into RAM. Peak memory for
X data during training is roughly batch_size * 5000 * 12 * 4 bytes (~15MB at
batch_size=64), regardless of dataset size.

Usage:
    python train.py --data_dir ./preprocessed_27178 --out_dir ./model_out_27178
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from resnet1d import ResNet1D34


def subject_level_split(subjects, y, val_size=0.10, test_size=0.20, seed=42):
    unique_subjects = np.unique(subjects)
    subj_has_pos = np.array([
        y[subjects == s].max() for s in unique_subjects
    ])
    train_val_subj, test_subj = train_test_split(
        unique_subjects, test_size=test_size, stratify=subj_has_pos, random_state=seed
    )
    train_val_pos = np.array([subj_has_pos[np.where(unique_subjects == s)[0][0]] for s in train_val_subj])
    val_frac_of_remaining = val_size / (1 - test_size)
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=val_frac_of_remaining, stratify=train_val_pos, random_state=seed
    )
    return set(train_subj), set(val_subj), set(test_subj)


class MemmapECGDataset(Dataset):
    """
    Reads individual ECG records on-demand from a memory-mapped X.npy file.
    `row_indices` are positions into the FULL X array (not into y_subset).
    """
    def __init__(self, x_npy_path, row_indices, y_subset):
        self.x_npy_path = x_npy_path
        self.row_indices = np.asarray(row_indices)
        self.y_subset = np.asarray(y_subset, dtype=np.float32)
        self._X = None  # opened lazily so each DataLoader worker gets its own handle

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.x_npy_path, mmap_mode='r')

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        self._ensure_open()
        row = self.row_indices[idx]
        x = np.array(self._X[row])  # copies just this one record from disk
        return torch.from_numpy(x).float(), torch.tensor(self.y_subset[idx], dtype=torch.float32)


def main(data_dir, out_dir, epochs=100, patience=10, batch_size=64, lr=1e-4, num_workers=0):
    os.makedirs(out_dir, exist_ok=True)

    x_npy_path = os.path.join(data_dir, 'X.npy')
    y = np.load(os.path.join(data_dir, 'y.npy'), allow_pickle=True)
    groups = np.load(os.path.join(data_dir, 'groups.npy'), allow_pickle=True)
    subjects = np.load(os.path.join(data_dir, 'subjects.npy'), allow_pickle=True)

    train_subj, val_subj, test_subj = subject_level_split(subjects, y)
    assert len(train_subj & val_subj) == 0
    assert len(train_subj & test_subj) == 0
    assert len(val_subj & test_subj) == 0

    train_mask = np.isin(subjects, list(train_subj))
    val_mask = np.isin(subjects, list(val_subj))
    test_mask = np.isin(subjects, list(test_subj))

    print(f"Split sizes -- train: {train_mask.sum()} records ({len(train_subj)} subjects), "
          f"val: {val_mask.sum()} records ({len(val_subj)} subjects), "
          f"test: {test_mask.sum()} records ({len(test_subj)} subjects)")

    train_rows = np.where(train_mask)[0]
    val_rows = np.where(val_mask)[0]
    y_train, y_val = y[train_mask], y[val_mask]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)

    train_ds = MemmapECGDataset(x_npy_path, train_rows, y_train)
    val_ds = MemmapECGDataset(x_npy_path, val_rows, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ResNet1D34().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = crit(out, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.savez(os.path.join(out_dir, 'split_indices.npz'),
             train_subj=list(train_subj), val_subj=list(val_subj), test_subj=list(test_subj))
    print(f"Saved best model (val_loss={best_val_loss:.4f}) to {out_dir}/model.pt")
    print("Run evaluate.py next, pointing --data_dir to the same preprocessed folder.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                     help='Folder containing X.npy, y.npy, groups.npy, subjects.npy (from preprocess.py)')
    ap.add_argument('--out_dir', default='./model_out')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=0,
                     help='0 is safest on Windows; try 2 if training is I/O bound and stable')
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.epochs, args.patience, args.batch_size, args.lr, args.num_workers)
