"""
=============================================================================
Cross-Attention-based Model エンドツーエンド スループット計測スクリプト
=============================================================================
目的:
  事前計算済みの特徴量(.npy)を使う test_add_volstab.py とは異なり、
  「未知の生音声(.wav)が入力された場合」を想定し、
  以下の処理をすべて含めた end-to-end のスループットを計測する。

    生のvocal音声(.wav)  --[wav2vec 2.0]-->  vocal特徴 (T, 768)
    生のBGM音声(.wav)    --[Librosa MFCC]-->  BGM特徴   (T, 64)
    vocal特徴 + BGM特徴  --[Cross-Attention-based Model]-->  FLAMEパラメータ (T, 56)

  e2e_throughput_concat.py (Concatenation-based Model版) と条件を揃えて
  実行することで、両モデルを公平に比較できる。

使い方:
  # デフォルト設定(20サンプル、10回試行、240フレーム、GPU)
  python e2e_throughput_crossattn.py

  # サンプル数・フレーム数を変えて比較(SingingHead論文との比較用)
  python e2e_throughput_crossattn.py --n_samples 1
  python e2e_throughput_crossattn.py --n_samples 30
  python e2e_throughput_crossattn.py --seq_len 480
=============================================================================
"""

import os
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


# ============================================================
# パス設定(実際の環境に合わせて変更してください)
# ============================================================
DATASET_BASE_DIR = Path("/home/nagao2/src/Visualization/data/SingingHead")
CHECKPOINT_DIR    = Path("/home/nagao2/src/Visualization/checkpoints")

# 生音声(.wav)が置かれているディレクトリ(Concatenation版と共通)
VOCAL_WAV_DIR = DATASET_BASE_DIR / "audio_seqs"
BGM_WAV_DIR   = DATASET_BASE_DIR / "bgm_seqs"

TEST_TXT = DATASET_BASE_DIR / "test.txt"

# Cross-Attention-based Model のチェックポイント(test_add_volstab.py と同じもの)
CROSSATTN_CKPT = CHECKPOINT_DIR / "crossattn_pe_volstab_best_model.pth"

WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
N_MFCC = 64

BGM_SUFFIX   = "_bgm.wav"
VOCAL_SUFFIX = ".wav"

EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6
TARGET_DIM      = EXP_ONLY_DIM + POSE_NO_JAW_DIM  # = 56


# ============================================================
# モデル定義(test_add_volstab.py と完全に同一)
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
# 特徴抽出(生波形 -> vocal特徴 / BGM特徴)
# e2e_throughput_concat.py の OnlineFeatureExtractor と同一処理
# ============================================================
class OnlineFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        print(f"Loading wav2vec 2.0 model: {WAV2VEC_MODEL_NAME}")
        self.w2v_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL_NAME)
        self.w2v_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_NAME).to(device)
        self.w2v_model.eval()

    @staticmethod
    def _resize_feature(feat: np.ndarray, target_len: int) -> np.ndarray:
        old_len = feat.shape[0]
        if old_len == target_len:
            return feat
        old_idx = np.linspace(0, old_len - 1, old_len)
        new_idx = np.linspace(0, old_len - 1, target_len)
        out = np.zeros((target_len, feat.shape[1]), dtype=np.float32)
        for d in range(feat.shape[1]):
            out[:, d] = np.interp(new_idx, old_idx, feat[:, d])
        return out

    def extract_vocal_feature(self, wav_path: str, seq_len: int) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        inputs = self.w2v_feature_extractor(
            wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features = self.w2v_model(**inputs).last_hidden_state  # (1, T_raw, 768)

        features = features.squeeze(0)  # (T_raw, 768)
        features = (
            F.interpolate(
                features.T.unsqueeze(0), size=seq_len,
                mode="linear", align_corners=False,
            ).squeeze(0).T
        )
        return features.unsqueeze(0)  # (1, seq_len, 768)

    def extract_bgm_feature(self, wav_path: str, seq_len: int) -> torch.Tensor:
        y, sr = librosa.load(wav_path, sr=None)
        hop_length = int(sr / 30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)
        mfcc = mfcc.T  # (T, 64)
        mfcc = self._resize_feature(mfcc, seq_len)
        mfcc_tensor = torch.from_numpy(mfcc.astype(np.float32))
        return mfcc_tensor.unsqueeze(0).to(self.device)  # (1, seq_len, 64)


# ============================================================
# パラメータ数カウント
# ============================================================
def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# エンドツーエンド推論時間計測
# ============================================================
def measure_e2e_throughput(
    model: nn.Module,
    extractor: OnlineFeatureExtractor,
    data_ids: list,
    device: torch.device,
    seq_len: int,
    n_runs_per_sample: int,
):
    model.eval()
    all_times = []
    skipped = []

    for data_id in data_ids:
        vocal_path = VOCAL_WAV_DIR / f"{data_id}{VOCAL_SUFFIX}"
        bgm_path   = BGM_WAV_DIR / f"{data_id}{BGM_SUFFIX}"

        if not vocal_path.exists() or not bgm_path.exists():
            skipped.append(data_id)
            continue

        # ---- warmup ----
        for _ in range(3):
            v_feat = extractor.extract_vocal_feature(str(vocal_path), seq_len)
            b_feat = extractor.extract_bgm_feature(str(bgm_path), seq_len)
            with torch.no_grad():
                _ = model(v_feat, b_feat)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ---- 計測 ----
        start = time.time()
        for _ in range(n_runs_per_sample):
            v_feat = extractor.extract_vocal_feature(str(vocal_path), seq_len)
            b_feat = extractor.extract_bgm_feature(str(bgm_path), seq_len)
            with torch.no_grad():
                _ = model(v_feat, b_feat)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        avg_time_this_sample = (end - start) / n_runs_per_sample
        all_times.append(avg_time_this_sample)

    if skipped:
        print(f"警告: {len(skipped)} 件のファイルが見つからずスキップされました。"
              f"(例: {skipped[:3]})")

    return np.mean(all_times), np.std(all_times), len(all_times)


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cross-Attention-based Model のエンドツーエンド スループット計測"
    )
    parser.add_argument("--n_samples", type=int, default=20,
                        help="計測に使用するテストサンプル数")
    parser.add_argument("--n_runs_per_sample", type=int, default=10,
                        help="各サンプルにつき何回計測するか")
    parser.add_argument("--seq_len", type=int, default=240,
                        help="出力フレーム数(SingingHead標準は240=8秒,30fps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="サンプル抽出のランダムシード"
                             "(Concatenation版と同じ値にすることで同一サンプルで比較可能)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda または cpu(省略時は自動判定)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print(f"Config: n_samples={args.n_samples}, "
          f"n_runs_per_sample={args.n_runs_per_sample}, "
          f"seq_len={args.seq_len}, seed={args.seed}")

    # ---- テストセットからサンプルIDをランダムに取得 ----
    assert TEST_TXT.exists(), f"test.txt が見つかりません: {TEST_TXT}"
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        all_test_ids = [line.strip() for line in f if line.strip()]

    random.seed(args.seed)
    sample_ids = random.sample(all_test_ids, min(args.n_samples, len(all_test_ids)))
    print(f"計測に使用するサンプル数: {len(sample_ids)} / {len(all_test_ids)}")
    print(f"選ばれたサンプル: {sample_ids}")

    # ---- 特徴抽出器の準備(wav2vec2.0のロード) ----
    extractor = OnlineFeatureExtractor(device)

    # ---- モデルのロード ----
    model = MusicToExpressionTransformer(
        voice_dim=768, bgm_dim=64, exp_dim=TARGET_DIM,
        d_model=256, nhead=4, num_layers=4,
    ).to(device)
    assert CROSSATTN_CKPT.exists(), f"チェックポイントが見つかりません: {CROSSATTN_CKPT}"
    state_dict = torch.load(str(CROSSATTN_CKPT), map_location=device)
    model.load_state_dict(state_dict)

    total, trainable = count_params(model)
    print(f"\n=== Cross-Attention-based Model ===")
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    size_mb = os.path.getsize(CROSSATTN_CKPT) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")

    # ---- エンドツーエンドのスループット計測 ----
    avg_time, std_time, n_used = measure_e2e_throughput(
        model, extractor, sample_ids, device,
        seq_len=args.seq_len, n_runs_per_sample=args.n_runs_per_sample,
    )

    throughput = args.seq_len / avg_time
    print(f"\n--- End-to-End (raw audio -> wav2vec2.0/MFCC -> model) ---")
    print(f"Inference time: {avg_time * 1000:.2f} +/- {std_time * 1000:.2f} ms/seq "
          f"(n={n_used} real test samples, {args.n_runs_per_sample} runs each)")
    print(f"Throughput: {throughput:.1f} fps")
    print(f"Real-time capable (>=30fps): {'Yes' if throughput >= 30 else 'No'}")

    print(f"\n[参考] SingingHead (UniSinger) 論文報告値: 50.849 fps")
    print(f"[参考] Think2Sing 論文報告値: 200+ fps (motion subtitle生成を除く)")


if __name__ == "__main__":
    main()