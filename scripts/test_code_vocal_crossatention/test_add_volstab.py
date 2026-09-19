"""
=============================================================================
CrossAttention + PositionalEncoding + VolStab Model Inference Script
=============================================================================
Input:
  - wav2vec_features/{id}.npy  (399, 768) -> interpolated to T x 768
  - mfcc_features/{id}.npy     (240, 64)

Output:
  - predictions/predictions_crossattn_pe_volstab/{id}.pkl
      {
          'shapecode': np.ndarray (1, 100)   <- taken from GT
          'expcodes':  np.ndarray (240, 50)  <- model prediction
          'posecodes': np.ndarray (240, 9)   <- global+neck are model predictions, jaw is reused from GT
      }

Usage:
  # Single file
  python test_crossattn_pe_volstab.py --id id15_3_1_3

  # Batch inference via a txt file
  python test_crossattn_pe_volstab.py --txt test.txt

  # Explicitly specify a checkpoint
  python test_crossattn_pe_volstab.py --txt test.txt --checkpoint path/to/crossattn_pe_volstab_best_model.pth
=============================================================================
"""

import os
import math
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ============================================================
# Dimension definitions
# ============================================================
EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6
TARGET_DIM      = EXP_ONLY_DIM + POSE_NO_JAW_DIM  # = 56


# ============================================================
# Model definition
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
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
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
# Model loading
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> MusicToExpressionTransformer:
    model = MusicToExpressionTransformer(
        voice_dim=768, bgm_dim=64, exp_dim=TARGET_DIM,
        d_model=256, nhead=4, num_layers=4,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {checkpoint_path}")
    return model


# ============================================================
# Single-sample inference
# ============================================================

def infer_one(
    data_id: str,
    wav2vec_dir: str,
    mfcc_dir: str,
    flame_dir: str,
    model: MusicToExpressionTransformer,
    device: torch.device,
    seq_len: int = 240,
) -> dict:

    # wav2vec features (interpolate from 399 frames to seq_len)
    voice_feat = torch.from_numpy(
        np.load(os.path.join(wav2vec_dir, f"{data_id}.npy"))
    ).float()
    voice_feat = (
        F.interpolate(
            voice_feat.T.unsqueeze(0),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T
    )  # (240, 768)

    # MFCC features
    mfcc = torch.from_numpy(
        np.load(os.path.join(mfcc_dir, f"{data_id}.npy"))
    ).float()
    if mfcc.size(0) > seq_len:
        mfcc = mfcc[:seq_len, :]
    elif mfcc.size(0) < seq_len:
        mfcc = F.pad(mfcc, (0, 0, 0, seq_len - mfcc.size(0)), "constant", 0)

    # Add batch dimension and move to device
    voice_feat = voice_feat.unsqueeze(0).to(device)  # (1, 240, 768)
    mfcc       = mfcc.unsqueeze(0).to(device)        # (1, 240, 64)

    # Inference (volume is only used for the loss calculation -> not needed at inference time)
    with torch.no_grad():
        pred = model(voice_feat, mfcc)  # (1, 240, 56)

    pred = pred.squeeze(0).cpu().numpy()  # (240, 56)

    # Split the 56 dimensions apart
    expcodes    = pred[:, :EXP_ONLY_DIM]                       # (240, 50)
    global_pose = pred[:, EXP_ONLY_DIM:EXP_ONLY_DIM + 3]      # (240, 3)
    neck_pose   = pred[:, EXP_ONLY_DIM + 3:EXP_ONLY_DIM + 6]  # (240, 3)

    # Reuse jaw from GT
    flame_base = os.path.join(flame_dir, data_id)
    if os.path.exists(f"{flame_base}_pose.npy"):
        # npy version
        pose_gt  = np.load(f"{flame_base}_pose.npy").astype(np.float32)
        jaw_pose = pose_gt[:, 6:9]  # (240, 3)
        shape    = np.load(f"{flame_base}_shape.npy").astype(np.float32) \
                   if os.path.exists(f"{flame_base}_shape.npy") \
                   else np.zeros((1, 100), dtype=np.float32)
    else:
        # pkl version (fallback)
        with open(os.path.join(flame_dir, f"{data_id}.pkl"), "rb") as f:
            flame_data = pickle.load(f)
        pose_key = "posecodes" if "posecodes" in flame_data else "pose"
        jaw_pose = np.array(flame_data[pose_key], dtype=np.float32)[:, 6:9]
        shape    = flame_data.get("shapecode", np.zeros((1, 100), dtype=np.float32))

    # Assemble posecodes in FLAME format [global(3), neck(3), jaw(3)]
    posecodes = np.concatenate(
        [global_pose, neck_pose, jaw_pose], axis=-1
    )  # (240, 9)

    return {
        "shapecode": shape.astype(np.float32),
        "expcodes":  expcodes.astype(np.float32),
        "posecodes": posecodes.astype(np.float32),
    }


# ============================================================
# Batch inference
# ============================================================

def run_inference(
    data_ids: list,
    wav2vec_dir: str,
    mfcc_dir: str,
    flame_dir: str,
    output_dir: str,
    model: MusicToExpressionTransformer,
    device: torch.device,
    seq_len: int = 240,
):
    os.makedirs(output_dir, exist_ok=True)
    errors = []

    for data_id in tqdm(data_ids, desc="Inference"):
        output_path = os.path.join(output_dir, f"{data_id}.pkl")

        if os.path.exists(output_path):
            continue

        try:
            result = infer_one(
                data_id, wav2vec_dir, mfcc_dir, flame_dir,
                model, device, seq_len
            )
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            errors.append((data_id, str(e)))
            print(f"\nERROR: {data_id}: {e}")

    print(f"\nDone: {len(data_ids) - len(errors)}/{len(data_ids)} completed")
    if errors:
        print(f"Errors ({len(errors)}):")
        for data_id, err in errors:
            print(f"  {data_id}: {err}")

    return errors


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CrossAttention + PE + VolStab Model Inference Script"
    )
    parser.add_argument("--id",  type=str, default=None,
                        help="Single sample ID (e.g. id15_3_1_3)")
    parser.add_argument("--txt", type=str, default=None,
                        help="txt file containing a list of IDs (e.g. test.txt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if omitted)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-set if omitted)")
    parser.add_argument("--seq_len", type=int, default=240)
    args = parser.parse_args()

    # ---- Path configuration ----
    current_dir      = os.path.dirname(os.path.abspath(__file__))
    dataset_base_dir = os.path.abspath(
        os.path.join(current_dir, "..", "..", "data", "SingingHead")
    )

    wav2vec_dir = os.path.join(dataset_base_dir, "wav2vec_features")
    mfcc_dir    = os.path.join(dataset_base_dir, "mfcc_features")
    flame_dir   = os.path.join(dataset_base_dir, "flame_npy")

    # Auto-detect checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            current_dir, "..", "..", "checkpoints",
            "crossattn_pe_volstab_best_model.pth"
        )
    assert os.path.exists(checkpoint_path), \
        f"Checkpoint not found: {checkpoint_path}"

    # Output destination
    output_dir = args.output_dir or os.path.join(
        dataset_base_dir, "predictions", "predictions_crossattn_pe_volstab"
    )

    # ---- Load device and model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(checkpoint_path, device)

    # ---- Collect target IDs for inference ----
    if args.id:
        data_ids = [args.id]
    elif args.txt:
        txt_path = args.txt if os.path.isabs(args.txt) \
                   else os.path.join(dataset_base_dir, args.txt)
        with open(txt_path, "r", encoding="utf-8") as f:
            data_ids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Please specify either --id or --txt")

    print(f"Inference targets: {len(data_ids)} samples")
    print(f"Output directory : {output_dir}")

    # ---- Run inference ----
    run_inference(
        data_ids, wav2vec_dir, mfcc_dir, flame_dir,
        output_dir, model, device, args.seq_len
    )

    # ---- Check a sample output ----
    sample_path = os.path.join(output_dir, f"{data_ids[0]}.pkl")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        print(f"\nSample check ({data_ids[0]}):")
        print(f"  shapecode : {sample['shapecode'].shape}")
        print(f"  expcodes  : {sample['expcodes'].shape}  "
              f"mean={sample['expcodes'].mean():.4f}")
        print(f"  posecodes : {sample['posecodes'].shape}  "
              f"mean={sample['posecodes'].mean():.4f}")


if __name__ == "__main__":
    main()
