import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_naft import NAFTMemmapDataset, NAFTResNet1D

class BalancedNAFTLoss(nn.Module):
    def __init__(self, num_groups=5, pos_weight=3.5, alpha=0.2, gamma=1.0):
        super(BalancedNAFTLoss, self).__init__()
        self.num_groups = num_groups
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.register_buffer('group_weights', torch.ones(num_groups) / num_groups)

    def forward(self, logits, targets, groups, feat_noisy, feat_clean=None):
        # 1. Weighted Cross Entropy (Positive Class Penalty)
        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[targets == 1] = self.pos_weight
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none') * weights

        # 2. Group DRO Loss
        group_losses = torch.zeros(self.num_groups, device=logits.device)
        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_losses[g] = ce_loss[mask].mean()

        with torch.no_grad():
            self.group_weights *= torch.exp(self.alpha * group_losses)
            self.group_weights /= self.group_weights.sum()

        loss_dro = torch.sum(self.group_weights * group_losses)

        # 3. Feature Invariance Loss
        loss_inv = 0.0
        if feat_clean is not None:
            loss_inv = 1.0 - F.cosine_similarity(feat_noisy, feat_clean, dim=-1).mean()

        return loss_dro + (self.gamma * loss_inv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--split_dir", type=str, default="./imbalanced_split")
    parser.add_argument("--out_dir", type=str, default="./model_naft_balanced")
    parser.add_argument("--pos_weight", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device} | pos_weight: {args.pos_weight}")

    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    groups_path = os.path.join(args.data_dir, "groups.npy")

    train_idx = np.load(os.path.join(args.split_dir, "train_imbalanced_idx.npy"))
    val_idx = np.load(os.path.join(args.split_dir, "val_idx.npy"))

    train_ds = NAFTMemmapDataset(x_path, y_path, groups_path, train_idx)
    val_ds = NAFTMemmapDataset(x_path, y_path, groups_path, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NAFTResNet1D(in_channels=1, num_classes=2).to(device)
    criterion = BalancedNAFTLoss(num_groups=5, pos_weight=args.pos_weight, gamma=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.out_dir, "model_naft_balanced.pt")

    print("\n--- Starting Balanced NAFT Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x_clean, yb, gb in train_loader:
            x_clean, yb, gb = x_clean.to(device), yb.to(device), gb.to(device)
            noise = torch.randn_like(x_clean) * 0.15
            x_noisy = x_clean + noise

            optimizer.zero_grad()
            logits, feat_noisy, feat_clean = model(x_noisy, x_clean)
            
            loss = criterion(logits, yb, gb, feat_noisy, feat_clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x_clean, yb, _ in val_loader:
                x_clean, yb = x_clean.to(device), yb.to(device)
                logits, _, _ = model(x_clean)
                val_loss += val_criterion(logits, yb).item() * len(yb)

        val_loss /= len(val_ds)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\n[SUCCESS] Balanced NAFT Model saved to {best_model_path}")

if __name__ == "__main__":
    main()