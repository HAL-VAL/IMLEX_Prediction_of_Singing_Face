"""
=============================================================================
Cross-Attention Audio+BGM モデル 推論スクリプト
=============================================================================
入力:
  - wav2vec_features/{id}.npy  (399, 768)
  - mfcc_features/{id}.npy     (240, 64)

出力:
  - predictions_crossattn/{id}.pkl
      {
          'shapecode': np.ndarray (1, 100)   ← GT から取得
          'expcodes':  np.ndarray (240, 50)
          'posecodes': np.ndarray (240, 9)   ← global+neck はモデル予測、jaw は GT 流用
      }

アーキテクチャの特徴（Audio+BGM concat版との違い）:
  - voice_projector: Linear(768 → 256)  ← d_model // 2 ではなく d_model フルサイズ
  - bgm_projector:   Linear(64  → 256)  ← 同上
  - Cross-Attention: Q=voice, K=V=bgm  →「BGMのリズムに合わせてvocalの注目箇所を算出」
  - その後 TransformerEncoder で時系列を学習

使い方:
  # 単一ファイル
  python test.py --id id15_3_1_3

  # txtファイルに書かれた全IDを一括推論
  python test.py --txt test.txt

  # チェックポイントを指定
  python test.py --txt test.txt --checkpoint path/to/crossattention_audio_bgm_best_model_ep50.pth
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
# 次元定義
# ============================================================
EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6
TARGET_DIM      = EXP_ONLY_DIM + POSE_NO_JAW_DIM  # = 56


# ============================================================
# モデル定義（Cross-Attention 版）
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MusicToExpressionTransformer(nn.Module):
    def __init__(self, voice_dim=768, bgm_dim=64, exp_dim=TARGET_DIM,
                 d_model=256, nhead=4, num_layers=4):
        super().__init__()
        # concat版と異なり、両方 d_model フルサイズへ射影
        self.voice_projector = nn.Linear(voice_dim, d_model)  # 768 → 256
        self.bgm_projector   = nn.Linear(bgm_dim,   d_model)  # 64  → 256

        # Cross-Attention: Q=voice, K=V=bgm
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=1024, dropout=0.1, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(d_model, exp_dim)

    def forward(self, voice_feat, bgm_feat):
        # 1. 特徴を d_model 次元に射影
        q = self.voice_projector(voice_feat)  # (B, T, 256)  Query = vocal
        k = v = self.bgm_projector(bgm_feat)  # (B, T, 256)  Key/Value = BGM

        # 2. Cross-Attention:「BGMのリズムに合わせてvocalのどの部分を重視するか」
        attn_out, _ = self.cross_attn(q, k, v)
        x = self.norm1(q + attn_out)          # 残差結合 + LayerNorm

        # 3. TransformerEncoder で時系列的な流れを学習
        x = self.transformer_encoder(x)
        return self.output_layer(x)


# ============================================================
# 推論関数
# ============================================================

def load_model(checkpoint_path: str, device: torch.device) -> MusicToExpressionTransformer:
    model = MusicToExpressionTransformer(
        voice_dim=768, bgm_dim=64, exp_dim=TARGET_DIM,
        d_model=256, nhead=4, num_layers=4,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"モデルをロードしました: {checkpoint_path}")
    return model


def infer_one(
    data_id: str,
    wav2vec_dir: str,
    mfcc_dir: str,
    flame_dir: str,
    model: MusicToExpressionTransformer,
    device: torch.device,
    seq_len: int = 240,
) -> dict:
    # wav2vec 特徴の読み込み（399フレーム → seq_len へ補間）
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

    # MFCC 特徴の読み込み（既に240フレーム固定）
    mfcc = torch.from_numpy(
        np.load(os.path.join(mfcc_dir, f"{data_id}.npy"))
    ).float()  # (240, 64)

    # 長さ保険
    if mfcc.size(0) > seq_len:
        mfcc = mfcc[:seq_len, :]
    elif mfcc.size(0) < seq_len:
        mfcc = F.pad(mfcc, (0, 0, 0, seq_len - mfcc.size(0)), "constant", 0)

    # バッチ次元を追加してGPUへ
    voice_feat = voice_feat.unsqueeze(0).to(device)  # (1, 240, 768)
    mfcc       = mfcc.unsqueeze(0).to(device)        # (1, 240, 64)

    # 推論
    with torch.no_grad():
        pred = model(voice_feat, mfcc)  # (1, 240, 56)

    pred = pred.squeeze(0).cpu().numpy()  # (240, 56)

    # 56次元を分解
    expcodes    = pred[:, :EXP_ONLY_DIM]                   # (240, 50)
    global_pose = pred[:, EXP_ONLY_DIM:EXP_ONLY_DIM + 3]  # (240, 3)
    neck_pose   = pred[:, EXP_ONLY_DIM + 3:]               # (240, 3)

    # jaw は元データの GT jaw を使用（モデルは jaw を予測しないため）
    original_pkl_path = os.path.join(flame_dir, f"{data_id}.pkl")
    with open(original_pkl_path, "rb") as f:
        flame_data = pickle.load(f)
    pose_key = "posecodes" if "posecodes" in flame_data else "pose"
    jaw_pose = np.array(flame_data[pose_key], dtype=np.float32)[:, 6:9]  # (240, 3)

    # posecodes を FLAME 形式 [global(3), neck(3), jaw(3)] に組み立て
    posecodes = np.concatenate([global_pose, neck_pose, jaw_pose], axis=-1)  # (240, 9)

    return {
        "shapecode": flame_data["shapecode"].astype(np.float32),
        "expcodes":  expcodes.astype(np.float32),
        "posecodes": posecodes.astype(np.float32),
    }


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
            result = infer_one(data_id, wav2vec_dir, mfcc_dir, flame_dir, model, device, seq_len)
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            errors.append((data_id, str(e)))
            print(f"\nERROR: {data_id}: {e}")

    print(f"\n完了: {len(data_ids) - len(errors)}/{len(data_ids)} 件")
    if errors:
        print(f"エラー ({len(errors)} 件): {errors}")

    return errors


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Audio+BGM モデル 推論スクリプト")
    parser.add_argument("--id",  type=str, default=None,
                        help="単一サンプルのID（例: id15_3_1_3）")
    parser.add_argument("--txt", type=str, default=None,
                        help="IDリストのtxtファイル（例: test.txt）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="チェックポイントのパス（省略時は自動検索）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="出力先ディレクトリ（省略時は自動設定）")
    parser.add_argument("--seq_len", type=int, default=240)
    args = parser.parse_args()

    # ---- パス設定 ----
    current_dir      = os.path.dirname(os.path.abspath(__file__))
    dataset_base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "SingingHead"))

    wav2vec_dir = os.path.join(dataset_base_dir, "wav2vec_features")
    mfcc_dir    = os.path.join(dataset_base_dir, "mfcc_features")
    flame_dir   = os.path.join(dataset_base_dir, "flame_seqs")

    # チェックポイントの自動検索
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            current_dir, "..", "..", "checkpoints",
            "crossattention_audio_bgm_best_model_ep50.pth"
        )
    assert os.path.exists(checkpoint_path), f"チェックポイントが見つかりません: {checkpoint_path}"

    # 出力先（他モデルと区別するため predictions_crossattn/ に保存）
    output_dir = args.output_dir or os.path.join(dataset_base_dir, "predictions/predictions_crossattn")

    # ---- デバイス・モデルのロード ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(checkpoint_path, device)

    # ---- 推論対象IDの収集 ----
    if args.id:
        data_ids = [args.id]
    elif args.txt:
        txt_path = args.txt if os.path.isabs(args.txt) else os.path.join(dataset_base_dir, args.txt)
        with open(txt_path, "r", encoding="utf-8") as f:
            data_ids = [line.strip() for line in f if line.strip()]
    else:
        parser.error("--id または --txt のどちらかを指定してください")

    print(f"推論対象: {len(data_ids)} 件")
    print(f"出力先:   {output_dir}")

    # ---- 推論実行 ----
    run_inference(data_ids, wav2vec_dir, mfcc_dir, flame_dir, output_dir, model, device, args.seq_len)

    # ---- 出力サンプルの確認 ----
    sample_id   = data_ids[0]
    sample_path = os.path.join(output_dir, f"{sample_id}.pkl")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        print(f"\nサンプル確認 ({sample_id}):")
        print(f"  shapecode shape: {sample['shapecode'].shape}")
        print(f"  expcodes shape:  {sample['expcodes'].shape}")
        print(f"  posecodes shape: {sample['posecodes'].shape}")
        print(f"  expcodes  mean:  {sample['expcodes'].mean():.4f}")
        print(f"  posecodes mean:  {sample['posecodes'].mean():.4f}")


if __name__ == "__main__":
    main()