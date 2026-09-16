"""
=============================================================================
SingingHead-Animation: Music-Driven Facial Expression Estimation
(Cross-Attention version)
=============================================================================
Model: MusicToExpressionTransformer (Cross-Attention approach)
Input:  Vocal wav2vec features (768dim) + BGM MFCC features (64dim)
        * Voice/BGM are each fully projected to d_model (256), and instead of
        a simple concat they are fused via Cross-Attention
        (Query=voice, Key/Value=bgm)
        Goal: learn "which parts of the voice to emphasize in sync with
        the BGM's rhythm"
        * PositionalEncoding (positional information) is NOT used here
        (it is added in e.g. train_add_volstab.py)
        The Cross-Attention output goes through a residual connection +
        LayerNorm before being passed to the TransformerEncoder (4 layers)
Output: FLAME expression + neck/global pose (56 dims total, jaw excluded)
Loss:   MSE + velocity loss (vel) only
        * Silence-stabilization loss (vol_stab) not implemented
Data:   Loaded from disk on every __getitem__ call
        (no RAM preloading; flame is .pkl format, volume is unused)
Training: epochs=50, batch_size=64, lr=1e-4, seq_len=240
Checkpoint: Overwritten and saved only when validation loss improves
            (keeps only the single best checkpoint)
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
import wandb
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import numpy as np


# ----------------------------------------------------
# 1. Model definition
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class MusicToExpressionTransformer(nn.Module):
        
    def __init__(self, voice_dim=768, bgm_dim=64, exp_dim=56, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.voice_projector = nn.Linear(voice_dim, d_model)
        self.bgm_projector = nn.Linear(bgm_dim, d_model)

        # Add cross-attention layer
        # Query: latent representation of expression parameters, Key/Value: voice/music features
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Layer for sequential processing (organizes the info after attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, exp_dim)

    def forward(self, voice_feat, bgm_feat):
            # 1. Project features into d_model dimensions
            q = self.voice_projector(voice_feat) # Use voice as Query
            k = v = self.bgm_projector(bgm_feat) # Use BGM as Key/Value
            
            # 2. Cross-attention: computes "which parts of the voice to
            # emphasize in sync with the BGM's rhythm"
            attn_out, _ = self.cross_attn(q, k, v)
            x = self.norm1(q + attn_out) # Residual connection
            
            # 3. Learn the temporal flow (change in expression) with the Transformer Encoder
            x = self.transformer_encoder(x)
            return self.output_layer(x)
        

def compute_loss(pred, target, lambda_vel=1.0):
    mse_loss = F.mse_loss(pred, target)
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    vel_loss = F.mse_loss(pred_vel, target_vel)
    return mse_loss + lambda_vel * vel_loss, mse_loss, vel_loss

# ----------------------------------------------------
# 2. Dataset definition
# ----------------------------------------------------
class RealSingingHeadDataset(Dataset):
    def __init__(self, txt_path, wav2vec_dir, mfcc_dir, flame_dir, seq_len=240):
        self.wav2vec_dir = wav2vec_dir
        self.mfcc_dir = mfcc_dir
        self.flame_dir = flame_dir
        self.seq_len = seq_len
        with open(txt_path, 'r', encoding='utf-8') as f:
            self.data_ids = [line.strip() for line in f if line.strip()]

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        mfcc_path = os.path.join(self.mfcc_dir, f"{data_id}.npy")
        flame_path = os.path.join(self.flame_dir, f"{data_id}.pkl")
        voice_path = os.path.join(self.wav2vec_dir, f"{data_id}.npy")

        # Load wav2vec features
        voice_feat = torch.from_numpy(np.load(voice_path)).float()
        voice_feat = (
            F.interpolate(
                voice_feat.T.unsqueeze(0),
                size=self.seq_len,
                mode="linear",
                align_corners=False
            )
            .squeeze(0)
            .T
        )
            
        # Load MFCC features
        mfcc_path = os.path.join(self.mfcc_dir, f"{data_id}.npy")
        mfcc = torch.from_numpy(np.load(mfcc_path)).float()


            
        # Load ground-truth expression data
        with open(flame_path, 'rb') as f:
            flame_data = pickle.load(f)
            
        exp_key = 'expcodes' if 'expcodes' in flame_data else 'expression'
        pose_key = 'posecodes' if 'posecodes' in flame_data else 'pose'
        exp = torch.FloatTensor(flame_data[exp_key])
        # Include only the non-jaw part of pose in the target
        # (jaw is also excluded from the prediction target)
        pose = torch.FloatTensor(flame_data[pose_key])
        pose_no_jaw = pose[:, :6]
        target_exp = torch.cat(
            [exp, pose_no_jaw],
            dim=-1
        )

        # Padding and truncation
        if mfcc.size(0) > self.seq_len: mfcc = mfcc[:self.seq_len, :]
        else: mfcc = F.pad(mfcc, (0, 0, 0, self.seq_len - mfcc.size(0)), "constant", 0)
            
        if target_exp.size(0) > self.seq_len: target_exp = target_exp[:self.seq_len, :]
        else: target_exp = F.pad(target_exp, (0, 0, 0, self.seq_len - target_exp.size(0)), "constant", 0)

        return (voice_feat, mfcc, target_exp)

# ----------------------------------------------------
# 3. Main training process
# ----------------------------------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "SingingHead"))

    train_txt = os.path.join(dataset_base_dir, "train.txt")
    val_txt = os.path.join(dataset_base_dir, "val.txt")
    mfcc_dir = os.path.join(dataset_base_dir, "mfcc_features")
    flame_dir = os.path.join(dataset_base_dir, "flame_seqs")
    wav2vec_dir = os.path.join(dataset_base_dir,"wav2vec_features")
    
    epochs = 50
    batch_size = 64
    lr = 1e-4
    seq_len = 240
    num_workers = 0
    
    start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if WANDB_AVAILABLE:
        wandb.init(
            project="SingingHead-Animation",
            name=f"crossattention_audio_bgm_{epochs}epochs_{start_time_str}",
            config={"epochs": epochs, "batch_size": batch_size, "lr": lr, "seq_len": seq_len}
        )
    
    train_dataset = RealSingingHeadDataset(train_txt, wav2vec_dir, mfcc_dir, flame_dir, seq_len=seq_len)
    val_dataset = RealSingingHeadDataset(val_txt, wav2vec_dir, mfcc_dir, flame_dir, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    sample_voice, sample_mfcc, sample_target = train_dataset[0]
    print(sample_voice.shape)
    print(sample_mfcc.shape)
    print(sample_target.shape)
    exp_dim = sample_target.shape[-1]
    
    model = MusicToExpressionTransformer(bgm_dim=64, exp_dim=exp_dim, d_model=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(next(model.parameters()).device)
    
    checkpoint_dir = os.path.join(current_dir, "..", "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Currently using device: {device}")
    print(f"\n--- Starting model training ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Train")
        for (voice_feat, mfcc, target_exp) in train_bar:
            t0 = time.time()
            voice_feat = voice_feat.to(device)
            mfcc = mfcc.to(device)
            target_exp = target_exp.to(device)
            
            optimizer.zero_grad()
            t1 = time.time()
            pred_exp = model(voice_feat, mfcc)
            loss, _, _ = compute_loss(pred_exp, target_exp)
            loss.backward()
            optimizer.step()

            t2 = time.time()
            #print("load:", t1-t0, "gpu:", t2-t1)
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch}/{epochs}] Val  ", leave=False)
        with torch.no_grad():
            for (voice_feat, mfcc, target_exp) in val_bar:
                voice_feat = voice_feat.to(device)
                mfcc = mfcc.to(device)
                target_exp = target_exp.to(device)
                pred_exp = model(voice_feat, mfcc)
                loss, _, _ = compute_loss(pred_exp, target_exp)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f" -> Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "best_val_loss": min(best_val_loss, avg_val_loss)})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            best_filename = "crossattention_audio_bgm_best_model_ep50.pth"
            best_path = os.path.join(checkpoint_dir, best_filename)
            torch.save(model.state_dict(), best_path)
            print(f"    * New best accuracy (Epoch {epoch})! Overwriting saved model: {best_path}")
            
            if WANDB_AVAILABLE:
                artifact = wandb.Artifact(name="audio-bgm-model", type="model", description=f"Achieved at epoch {epoch}")
                artifact.add_file(best_path)
                wandb.log_artifact(artifact)

    if WANDB_AVAILABLE: wandb.finish()

if __name__ == "__main__":
    main()
