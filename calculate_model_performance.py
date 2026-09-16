"""
=============================================================================
モデル効率性(パラメータ数・モデルサイズ・推論時間・スループット)計測スクリプト
実データ(テストセット)を用いて Baseline / Ours を比較する
=============================================================================
"""

import os
import time
import random
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# パス設定(Linux実パス)
# ============================================================
BASE_DIR = Path("/home/nagao2/src/Visualization")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BASELINE_FILE = BASE_DIR / "scripts" / "train_code_wav2andMFCC" / "train.py"
OURS_FILE     = BASE_DIR / "scripts" / "train_code_crossattention" / "train_add_volstab.py"

BASELINE_CKPT = CHECKPOINT_DIR / "audio_bgm_best_model_ep100.pth"
OURS_CKPT     = CHECKPOINT_DIR / "crossattn_pe_volstab_best_model.pth"

# データセットのパス(train.py / test_crossattn_pe_volstab.py と同じ構成)
DATASET_BASE_DIR = BASE_DIR / "data" / "SingingHead"
WAV2VEC_DIR = DATASET_BASE_DIR / "wav2vec_features"
MFCC_DIR    = DATASET_BASE_DIR / "mfcc_features"
TEST_TXT    = DATASET_BASE_DIR / "test.txt"

SEQ_LEN = 240
N_SAMPLES = 20          # 計測に使うテストサンプル数(平均を取る)
N_RUNS_PER_SAMPLE = 10  # 各サンプルにつき何回forwardを回すか
RANDOM_SEED = 42        # 再現性のための固定シード


# ============================================================
# モジュール読み込み(ファイルパス指定でロード、名前衝突を回避)
# ============================================================
def load_module_from_path(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_module = load_module_from_path("baseline_train", BASELINE_FILE)
ours_module     = load_module_from_path("ours_train", OURS_FILE)


# ============================================================
# 実データの読み込み(test_crossattn_pe_volstab.py の infer_one と同じロジック)
# ============================================================
def load_real_sample(data_id: str, seq_len: int = SEQ_LEN):
    # wav2vec 特徴(399フレーム → seq_len へ補間)
    voice_feat = torch.from_numpy(
        np.load(os.path.join(WAV2VEC_DIR, f"{data_id}.npy"))
    ).float()
    voice_feat = (
        F.interpolate(
            voice_feat.T.unsqueeze(0),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T
    )  # (seq_len, 768)

    # MFCC 特徴
    mfcc = torch.from_numpy(
        np.load(os.path.join(MFCC_DIR, f"{data_id}.npy"))
    ).float()
    if mfcc.size(0) > seq_len:
        mfcc = mfcc[:seq_len, :]
    elif mfcc.size(0) < seq_len:
        mfcc = F.pad(mfcc, (0, 0, 0, seq_len - mfcc.size(0)), "constant", 0)

    return voice_feat.unsqueeze(0), mfcc.unsqueeze(0)  # (1, seq_len, dim)


# ============================================================
# パラメータ数カウント
# ============================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 推論時間計測(実データ・複数サンプル平均)
# ============================================================
def measure_inference_real_data(model, data_ids, device, seq_len=SEQ_LEN,
                                  n_runs_per_sample=N_RUNS_PER_SAMPLE):
    model.eval()
    all_times = []

    with torch.no_grad():
        for data_id in data_ids:
            voice_feat, mfcc = load_real_sample(data_id, seq_len=seq_len)
            voice_feat = voice_feat.to(device)
            mfcc = mfcc.to(device)

            # warmup(このサンプルで数回捨てる)
            for _ in range(3):
                _ = model(voice_feat, mfcc)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            for _ in range(n_runs_per_sample):
                _ = model(voice_feat, mfcc)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            avg_time_this_sample = (end - start) / n_runs_per_sample
            all_times.append(avg_time_this_sample)

    return np.mean(all_times), np.std(all_times)


# ============================================================
# モデルごとのプロファイル実行
# ============================================================
def profile_model(name, model, ckpt_path, data_ids, device, seq_len=SEQ_LEN):
    print(f"\n=== {name} ({ckpt_path.name}) ===")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 1. パラメータ数
    total, trainable = count_params(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    # 2. ストレージサイズ
    size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")

    # 3. 推論時間・スループット(実データ、複数サンプル平均)
    avg_time, std_time = measure_inference_real_data(
        model, data_ids, device, seq_len=seq_len
    )
    throughput = seq_len / avg_time
    print(f"Inference time: {avg_time * 1000:.2f} ± {std_time * 1000:.2f} ms/seq "
          f"(n={len(data_ids)} real test samples, {N_RUNS_PER_SAMPLE} runs each)")
    print(f"Throughput: {throughput:.1f} fps")

    # 4. リアルタイム判定(SingingHeadは30fps)
    print(f"Real-time capable: {'Yes' if throughput >= 30 else 'No'}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- テストセットからサンプルIDをランダムに取得 ----
    assert TEST_TXT.exists(), f"test.txt が見つかりません: {TEST_TXT}"
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        all_test_ids = [line.strip() for line in f if line.strip()]

    random.seed(RANDOM_SEED)
    sample_ids = random.sample(all_test_ids, min(N_SAMPLES, len(all_test_ids)))

    print(f"計測に使用するサンプル数: {len(sample_ids)} / {len(all_test_ids)} "
          f"(random seed={RANDOM_SEED})")
    print(f"選ばれたサンプル: {sample_ids}")

    # ---- Baseline ----
    baseline_model = baseline_module.MusicToExpressionTransformer(
        bgm_dim=64, exp_dim=56, d_model=256
    )
    profile_model("Baseline", baseline_model, BASELINE_CKPT, sample_ids, device)

    # ---- Ours ----
    ours_model = ours_module.MusicToExpressionTransformer(
        voice_dim=768, bgm_dim=64, exp_dim=ours_module.TARGET_DIM,
        d_model=256, nhead=4, num_layers=4
    )
    profile_model("Ours", ours_model, OURS_CKPT, sample_ids, device)