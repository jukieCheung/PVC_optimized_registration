#!/usr/bin/env python3
"""
DeepPVC (reproduction-oriented) training code on axial slice triplets.

Inputs
------
- A folder produced by `export_axial_slices_triplet.py`, e.g. `/mnt/data/deeppvc_slices`:
    deeppvc_slices/
      <pair_id>/
        mr/slice_000.npy
        pet/slice_000.npy
        rbv/slice_000.npy
      index.csv   # columns: pair_id, slice_idx, z_mm, mr_path, pet_path, rbv_path

What this script does
---------------------
- Builds subject-level train/val split (no subject leakage) from index.csv
- 2‑channel input (MR+PET), 1‑channel target (RBV)
- 2D U‑Net (3 down-samples) with BN+ReLU; MSE loss
- Adam(β1=0.723, β2=0.999), lr=1.8e-3 with linear decay to 0 over `--epochs`
- Augmentation: random rotation (±30°) + horizontal flip (apply the same transform
  to MR/PET/RBV consistently)
- Tracks loss and SSIM on validation; saves best checkpoint by val SSIM

Usage
-----
python deeppvc_train.py \
  --root /mnt/data/deeppvc_slices \
  --epochs 40 --batch 16 --val-frac 0.2 --seed 42 \
  --num-workers 4 --out /mnt/data/deeppvc_ckpt

Notes
-----
- For full paper-level training you can set `--epochs 400`; start small to verify.
- If you prepared slices with per-volume mean normalization, keep it. Otherwise,
  you can add per-slice normalization in the Dataset below.
- ICC (region-level) needs VOI masks; this script reports MSE/SSIM at slice-level.
"""

import argparse
import os
from pathlib import Path
import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from skimage.metrics import structural_similarity as ssim_np
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# ----------------------------------------------------
#                 Dataset / Augment
# ----------------------------------------------------

class SliceTripletDataset(Dataset):
    def __init__(self, rows, augment=False, per_slice_norm=False):
        self.rows = rows
        self.augment = augment
        self.per_slice_norm = per_slice_norm

    def __len__(self):
        return len(self.rows)

    def _load_slice(self, path):
        arr = np.load(path)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0)
        if self.per_slice_norm:
            m = float(arr.mean())
            if m != 0:
                arr = arr / m
        return arr.astype(np.float32)

    def _rand_affine(self, H, W):
        # rotation in degrees
        angle = random.uniform(-30.0, 30.0)
        flip = random.random() < 0.5
        # build theta for affine_grid: rotation around center
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32)  # 2x3
        return theta, flip

    def _apply_aug(self, x, theta, flip):
        # x: (C,H,W) torch tensor
        C, H, W = x.shape
        grid = F.affine_grid(theta.unsqueeze(0), size=(1, C, H, W), align_corners=False)
        x = x.unsqueeze(0)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        x = x.squeeze(0)
        if flip:
            x = torch.flip(x, dims=[2])  # horizontal flip (W)
        return x

    def __getitem__(self, idx):
        r = self.rows[idx]
        mr = self._load_slice(r['mr_path'])  # (H,W)
        pet = self._load_slice(r['pet_path'])
        rbv = self._load_slice(r['rbv_path']) if r['rbv_path'] else None
        if rbv is None:
            # if GT missing, use zeros (won't be selected for train by default)
            rbv = np.zeros_like(mr, dtype=np.float32)

        # stack to tensors
        mr_t = torch.from_numpy(mr)[None, ...]   # (1,H,W)
        pet_t = torch.from_numpy(pet)[None, ...]
        x = torch.cat([mr_t, pet_t], dim=0)      # (2,H,W)
        y = torch.from_numpy(rbv)[None, ...]     # (1,H,W)

        if self.augment:
            theta, flip = self._rand_affine(*mr.shape)
            x = self._apply_aug(x, theta, flip)
            y = self._apply_aug(y, theta, flip)

        return x, y

# ----------------------------------------------------
#                        U-Net
# ----------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )
    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()
        # encoder
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.Conv2d(base, base, 2, 2)  # stride-2 conv (matches paper's strided conv)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.Conv2d(base*2, base*2, 2, 2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.Conv2d(base*4, base*4, 2, 2)

        # bottleneck
        self.mid = DoubleConv(base*4, base*8)

        # decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        m = self.mid(p3)

        u3 = self.up3(m)
        c3 = torch.cat([u3, d3], dim=1)
        d3 = self.dec3(c3)
        u2 = self.up2(d3)
        c2 = torch.cat([u2, d2], dim=1)
        d2 = self.dec2(c2)
        u1 = self.up1(d2)
        c1 = torch.cat([u1, d1], dim=1)
        d1 = self.dec1(c1)
        out = self.out(d1)
        return out

# ----------------------------------------------------
#                    Metrics / Utils
# ----------------------------------------------------

def ssim_batch(pred, target):
    """Compute mean SSIM per-batch using skimage (CPU)."""
    if not _HAS_SKIMAGE:
        return float('nan')
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    B = pred.shape[0]
    vals = []
    for i in range(B):
        x = pred[i, 0]
        y = target[i, 0]
        try:
            v = ssim_np(x, y, data_range=float(np.max(y) - np.min(y)) if np.max(y)>np.min(y) else 1.0)
        except Exception:
            v = np.nan
        vals.append(v)
    return float(np.nanmean(vals))

# ----------------------------------------------------
#                     Split / Loader
# ----------------------------------------------------

def load_index(root: Path):
    idx = pd.read_csv(root/"index.csv")
    # filter missing
    idx = idx[idx['mr_path'].notna() & idx['pet_path'].notna()]
    if 'rbv_path' in idx.columns:
        idx = idx[idx['rbv_path'].notna() & (idx['rbv_path']!="")]
    else:
        raise SystemExit("index.csv must contain 'rbv_path'.")
    # derive subject from pair_id (first token)
    idx['subject'] = idx['pair_id'].astype(str).str.split('/').str[0]
    return idx

def split_by_subject(idx_df, val_frac=0.2, seed=42):
    subs = sorted(idx_df['subject'].unique())
    rnd = random.Random(seed)
    rnd.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val_subs = set(subs[:n_val])
    train_rows = idx_df[~idx_df['subject'].isin(val_subs)].to_dict('records')
    val_rows = idx_df[idx_df['subject'].isin(val_subs)].to_dict('records')
    return train_rows, val_rows, val_subs

# ----------------------------------------------------
#                        Train
# ----------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    ssim_vals = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * x.size(0)
            ssim_vals.append(ssim_batch(pred, y))
    mean_loss = total_loss / len(loader.dataset)
    mean_ssim = float(np.nanmean(ssim_vals))
    return mean_loss, mean_ssim

# ----------------------------------------------------
#                        Main
# ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='deeppvc_slices root (folder containing index.csv)')
    ap.add_argument('--out', required=True, help='output folder for checkpoints/logs')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1.8e-3)
    ap.add_argument('--beta1', type=float, default=0.723)
    ap.add_argument('--beta2', type=float, default=0.999)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--per-slice-norm', action='store_true', help='divide each slice by its mean (if not already done)')
    args = ap.parse_args()

    root = Path(args.root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # data split
    idx_df = load_index(root)
    train_rows, val_rows, val_subs = split_by_subject(idx_df, val_frac=args.val_frac, seed=args.seed)

    train_ds = SliceTripletDataset(train_rows, augment=True, per_slice_norm=args.per_slice_norm)
    val_ds   = SliceTripletDataset(val_rows, augment=False, per_slice_norm=args.per_slice_norm)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model/opt/sched
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    # linear decay to zero over epochs
    lr_lambda = lambda e: max(0.0, 1.0 - (e / max(1, args.epochs)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    best_ssim = -1.0
    ckpt_path = out/'best.pth'

    log_rows = []
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_ld, opt, device)
        va_loss, va_ssim = eval_one_epoch(model, val_ld, device)
        sched.step()
        print(f"Epoch {epoch:03d} | train MSE {tr_loss:.6f} | val MSE {va_loss:.6f} | val SSIM {va_ssim:.4f} | lr {sched.get_last_lr()[0]:.6g}")
        log_rows.append({'epoch':epoch, 'train_mse':tr_loss, 'val_mse':va_loss, 'val_ssim':va_ssim, 'lr':sched.get_last_lr()[0]})
        # save best by SSIM
        if va_ssim > best_ssim:
            best_ssim = va_ssim
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_ssim': va_ssim}, ckpt_path)

    pd.DataFrame(log_rows).to_csv(out/'train_log.csv', index=False)
    print(f"[DONE] Best val SSIM={best_ssim:.4f}. Saved: {ckpt_path}")

if __name__ == '__main__':
    main()
