"""
=============================================================================
SingingHead-Animation: Music-Driven Facial Expression Estimation
(CrossAttention + PositionalEncoding + Volume-Stabilization version)
=============================================================================
Model: MusicToExpressionTransformer (Cross-Attention + PositionalEncoding)
Input:  Vocal wav2vec features (768dim) + BGM MFCC features (64dim)
        * Voice/BGM are each fully projected to d_model (256).
        * Unlike the earlier CrossAttention-only version, PositionalEncoding
          is now applied separately to both the voice (Query) and BGM
          (Key/Value) projections before attention.
        * Fusion is done via Cross-Attention (Query=voice, Key/Value=bgm),
          with a residual connection + LayerNorm before the
          TransformerEncoder (4 layers).
Output: FLAME expression + neck/global pose (56 dims total: 50 expression
        dims + 6 non-jaw pose dims; jaw is excluded)
        * Dimension layout is explicit via named constants:
          EXP_ONLY_DIM=50, POSE_NO_JAW_DIM=6, TARGET_DIM=56
          GLOBAL_POSE_START=50, NECK_POSE_START=53, NECK_POSE_END=56
Loss:   MSE + velocity loss (vel) + silence-stabilization loss (vol_stab)
        * vol_stab uses precomputed per-frame volume to down-weight
          movement during silent/low-volume segments, discouraging
          spurious expression jitter when there's no audio signal.
Data:   RAM-preloaded dataset (RealSingingHeadDataset)
        * All samples (voice/mfcc/flame/volume) are loaded into memory once
          at dataset construction time, so there's no disk I/O from the
          2nd epoch onward.
        * FLAME data is read from separate "_exp.npy" / "_pose.npy" files
          (flame_npy dir), not from .pkl as in earlier scripts.
        * Requires a precomputed volume_features directory alongside the
          usual wav2vec/mfcc/flame feature directories.
Training: epochs=50, batch_size=64, lr=1e-4, seq_len=240,
          lambda_vel=1.0, lambda_vol_stab=1.0, beta=5.0
Checkpoint: Overwritten and saved only when validation loss improves
            (keeps only the single best checkpoint:
            crossattn_pe_volstab_best_model.pth)
=============================================================================
"""
import os
import pickle
import math
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import numpy as np


# ============================================================
# Dimension definitions
# ============================================================
EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6
TARGET_DIM      = EXP_ONLY_DIM + POSE_NO_JAW_DIM  # = 56

GLOBAL_POSE_START = EXP_ONLY_DIM      # = 50
NECK_POSE_START   = EXP_ONLY_DIM + 3  # = 53
NECK_POSE_END     = EXP_ONLY_DIM + 6  # = 56


# ============================================================
# 1. Model definition (CrossAttention + PositionalEncoding)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MusicToExpressionTransformer(nn.Module):
    def __init__(
        self,
        voice_dim=768,
        bgm_dim=64,
        exp_dim=TARGET_DIM,
        d_model=256,
        nhead=4,
        num_layers=4,
    ):
        super().__init__()
        self.voice_projector   = nn.Linear(voice_dim, d_model)
        self.bgm_projector     = nn.Linear(bgm_dim,   d_model)
        self.pos_encoder_voice = PositionalEncoding(d_model)
        self.pos_encoder_bgm   = PositionalEncoding(d_model)
        self.cross_attn        = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, dropout=0.1, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, exp_dim)

    def forward(self, voice_feat, bgm_feat):
        q = self.voice_projector(voice_feat)
        k = self.bgm_projector(bgm_feat)
        v = k
        q = self.pos_encoder_voice(q)
        k = self.pos_encoder_bgm(k)
        v = k
        attn_out, _ = self.cross_attn(q, k, v)
        x = self.norm1(q + attn_out)
        x = self.transformer_encoder(x)
        return self.output_layer(x)


# ============================================================
# 2. Loss function (MSE + velocity loss + silence-stabilization loss)
# ============================================================

def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    volume: torch.Tensor,
    lambda_vel: float = 1.0,
    lambda_vol_stab: float = 1.0,
    beta: float = 5.0,
):
    # MSE loss
    mse_loss = F.mse_loss(pred, target)

    # Velocity loss (temporal smoothing)
    pred_vel   = pred[:, 1:, :]   - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    vel_loss   = F.mse_loss(pred_vel, target_vel)

    # Silence-stabilization loss (suppresses unwanted movement during silent segments)
    w_vol_pair     = torch.exp(-beta * 0.5 * (volume[:, 1:] + volume[:, :-1]))
    pred_diff_sq   = (pred[:, 1:, :] - pred[:, :-1, :]) ** 2
    pred_diff_norm = pred_diff_sq.mean(dim=-1)
    vol_stab_loss  = (w_vol_pair * pred_diff_norm).mean()

    total_loss = mse_loss + lambda_vel * vel_loss + lambda_vol_stab * vol_stab_loss

    loss_dict = {
        "total":    total_loss.item(),
        "mse":      mse_loss.item(),
        "vel":      vel_loss.item(),
        "vol_stab": vol_stab_loss.item(),
    }
    return total_loss, loss_dict


# ============================================================
# 3. Dataset
# ============================================================

class RealSingingHeadDataset(Dataset):
    """Loads all data into RAM at startup. No disk access from the 2nd epoch onward."""

    def __init__(
        self,
        txt_path: str,
        wav2vec_dir: str,
        mfcc_dir: str,
        flame_dir: str,
        volume_dir: str,
        seq_len: int = 240,
    ):
        self.wav2vec_dir = wav2vec_dir
        self.mfcc_dir    = mfcc_dir
        self.flame_dir   = flame_dir
        self.volume_dir  = volume_dir
        self.seq_len     = seq_len

        with open(txt_path, "r", encoding="utf-8") as f:
            self.data_ids = [line.strip() for line in f if line.strip()]

        self._preload_all()

    def __len__(self):
        return len(self.data_ids)

    def _preload_all(self):
        print(f"Preloading data into RAM... ({len(self.data_ids)} items)")
        self.cache = []

        for data_id in tqdm(self.data_ids, desc="Preloading"):

            # wav2vec features (interpolate from 399 frames to seq_len)
            voice_feat = torch.from_numpy(
                np.load(os.path.join(self.wav2vec_dir, f"{data_id}.npy"))
            ).float()
            voice_feat = (
                F.interpolate(
                    voice_feat.T.unsqueeze(0),
                    size=self.seq_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T
            )

            # MFCC features (fixed at 240 frames)
            mfcc = torch.from_numpy(
                np.load(os.path.join(self.mfcc_dir, f"{data_id}.npy"))
            ).float()
            if mfcc.size(0) > self.seq_len:
                mfcc = mfcc[: self.seq_len, :]
            elif mfcc.size(0) < self.seq_len:
                mfcc = F.pad(mfcc, (0, 0, 0, self.seq_len - mfcc.size(0)))

            # FLAME parameters (npy version)
            flame_base = os.path.join(self.flame_dir, data_id)
            exp  = torch.from_numpy(np.load(f"{flame_base}_exp.npy")).float()
            pose = torch.from_numpy(np.load(f"{flame_base}_pose.npy")).float()
            target_exp = torch.cat([exp, pose[:, :POSE_NO_JAW_DIM]], dim=-1)
            if target_exp.size(0) > self.seq_len:
                target_exp = target_exp[: self.seq_len, :]
            elif target_exp.size(0) < self.seq_len:
                target_exp = F.pad(target_exp, (0, 0, 0, self.seq_len - target_exp.size(0)))

            # Volume (precomputed)
            volume = torch.from_numpy(
                np.load(os.path.join(self.volume_dir, f"{data_id}.npy"))
            ).float()

            self.cache.append((voice_feat, mfcc, target_exp, volume))

        print("Loading complete.")

    def __getitem__(self, idx):
        return self.cache[idx]


# ============================================================
# 4. Main training loop
# ============================================================

def main():
    current_dir      = os.path.dirname(os.path.abspath(__file__))
    dataset_base_dir = os.path.abspath(
        os.path.join(current_dir, "..", "..", "data", "SingingHead")
    )

    train_txt   = os.path.join(dataset_base_dir, "train.txt")
    val_txt     = os.path.join(dataset_base_dir, "val.txt")
    mfcc_dir    = os.path.join(dataset_base_dir, "mfcc_features")
    flame_dir   = os.path.join(dataset_base_dir, "flame_npy")
    wav2vec_dir = os.path.join(dataset_base_dir, "wav2vec_features")
    volume_dir  = os.path.join(dataset_base_dir, "volume_features")

    # ---- Training hyperparameters ----
    epochs          = 50
    batch_size      = 64
    lr              = 1e-4
    seq_len         = 240
    lambda_vel      = 1.0
    lambda_vol_stab = 1.0
    beta            = 5.0

    start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if WANDB_AVAILABLE:
        wandb.init(
            project="SingingHead-Animation",
            name=f"CrossAttn-PE-VolStab-{epochs}ep-{start_time_str}",
            config={
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "seq_len": seq_len, "lambda_vel": lambda_vel,
                "lambda_vol_stab": lambda_vol_stab, "beta": beta,
                "model": "CrossAttn+PE+VolStab",
            },
        )

    # ---- Load data (RAM preload) ----
    train_dataset = RealSingingHeadDataset(
        train_txt, wav2vec_dir, mfcc_dir, flame_dir, volume_dir, seq_len=seq_len
    )
    val_dataset = RealSingingHeadDataset(
        val_txt, wav2vec_dir, mfcc_dir, flame_dir, volume_dir, seq_len=seq_len
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ---- Sanity-check the model ----
    sample_voice, sample_mfcc, sample_target, sample_volume = train_dataset[0]
    print(f"voice_feat : {sample_voice.shape}")
    print(f"mfcc       : {sample_mfcc.shape}")
    print(f"target_exp : {sample_target.shape}")
    print(f"volume     : {sample_volume.shape}")
    assert sample_target.shape[-1] == TARGET_DIM

    model = MusicToExpressionTransformer(
        voice_dim=768, bgm_dim=64, exp_dim=TARGET_DIM,
        d_model=256, nhead=4, num_layers=4,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice       : {device}")
    print(f"Total params : {total_params:,}")
    print(f"\n{'='*60}")
    print("Starting CrossAttention + PositionalEncoding + VolStab training")
    print(f"{'='*60}\n")

    checkpoint_dir = os.path.join(current_dir, "..", "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Train")

        for voice_feat, mfcc, target_exp, volume in train_bar:
            voice_feat = voice_feat.to(device)
            mfcc       = mfcc.to(device)
            target_exp = target_exp.to(device)
            volume     = volume.to(device)

            optimizer.zero_grad()
            pred_exp = model(voice_feat, mfcc)
            loss, loss_dict = compute_loss(
                pred_exp, target_exp, volume,
                lambda_vel=lambda_vel,
                lambda_vol_stab=lambda_vol_stab,
                beta=beta,
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_bar.set_postfix({
                "loss":     f"{loss.item():.4f}",
                "vol_stab": f"{loss_dict['vol_stab']:.4f}",
            })

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch}/{epochs}] Val  ", leave=False)

        with torch.no_grad():
            for voice_feat, mfcc, target_exp, volume in val_bar:
                voice_feat = voice_feat.to(device)
                mfcc       = mfcc.to(device)
                target_exp = target_exp.to(device)
                volume     = volume.to(device)

                pred_exp = model(voice_feat, mfcc)
                loss, _ = compute_loss(
                    pred_exp, target_exp, volume,
                    lambda_vel=lambda_vel,
                    lambda_vol_stab=lambda_vol_stab,
                    beta=beta,
                )
                val_loss_sum += loss.item()

        avg_train = train_loss_sum / len(train_loader)
        avg_val   = val_loss_sum   / len(val_loader)
        print(f" → Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
                "best_val_loss": min(best_val_loss, avg_val),
            })

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(
                checkpoint_dir, "crossattn_pe_volstab_best_model.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"    New best accuracy (Epoch {epoch}) | Val Loss: {avg_val:.4f}")
            print(f"       Saved to: {best_path}")

            if WANDB_AVAILABLE:
                artifact = wandb.Artifact(
                    name="crossattn-pe-volstab-model", type="model",
                    description=f"Best at epoch {epoch}, val_loss={avg_val:.4f}",
                )
                artifact.add_file(best_path)
                wandb.log_artifact(artifact)

    print(f"\n{'='*60}")
    print(f"Training complete | Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
