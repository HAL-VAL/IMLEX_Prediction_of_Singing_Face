"""
=============================================================================
Model Efficiency Measurement Script
(Parameter Count, Model Size, Inference Time, Throughput)
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
# Path settings
# ============================================================
BASE_DIR = Path("/home/nagao2/src/Visualization")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BASELINE_FILE = BASE_DIR / "scripts" / "train_code_wav2andMFCC" / "train.py"
OURS_FILE     = BASE_DIR / "scripts" / "train_code_crossattention" / "train_add_volstab.py"

BASELINE_CKPT = CHECKPOINT_DIR / "audio_bgm_best_model_ep100.pth"
OURS_CKPT     = CHECKPOINT_DIR / "crossattn_pe_volstab_best_model.pth"

DATASET_BASE_DIR = BASE_DIR / "data" / "SingingHead"
WAV2VEC_DIR = DATASET_BASE_DIR / "wav2vec_features"
MFCC_DIR    = DATASET_BASE_DIR / "mfcc_features"
TEST_TXT    = DATASET_BASE_DIR / "test.txt"

SEQ_LEN = 240
N_SAMPLES = 20          # Number of test samples used for measurement (averaged)
N_RUNS_PER_SAMPLE = 10  # Number of forward passes per sample
RANDOM_SEED = 42        # Fixed seed for reproducibility


# ============================================================
# Module loading
# ============================================================
def load_module_from_path(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_module = load_module_from_path("baseline_train", BASELINE_FILE)
ours_module     = load_module_from_path("ours_train", OURS_FILE)


# ============================================================
# Data loading
# ============================================================
def load_real_sample(data_id: str, seq_len: int = SEQ_LEN):
    # wav2vec features (interpolate 399 frames -> seq_len)
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

    # MFCC features
    mfcc = torch.from_numpy(
        np.load(os.path.join(MFCC_DIR, f"{data_id}.npy"))
    ).float()
    if mfcc.size(0) > seq_len:
        mfcc = mfcc[:seq_len, :]
    elif mfcc.size(0) < seq_len:
        mfcc = F.pad(mfcc, (0, 0, 0, seq_len - mfcc.size(0)), "constant", 0)

    return voice_feat.unsqueeze(0), mfcc.unsqueeze(0)  # (1, seq_len, dim)


# ============================================================
# Parameter count
# ============================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# Inference time measurement (real data, averaged over multiple samples)
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

            # warmup
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
# Run profiling for each model
# ============================================================
def profile_model(name, model, ckpt_path, data_ids, device, seq_len=SEQ_LEN):
    print(f"\n=== {name} ({ckpt_path.name}) ===")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 1. Parameter count
    total, trainable = count_params(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    # 2. Storage size
    size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")

    # 3. Inference time / throughput (real data, averaged over multiple samples)
    avg_time, std_time = measure_inference_real_data(
        model, data_ids, device, seq_len=seq_len
    )
    throughput = seq_len / avg_time
    print(f"Inference time: {avg_time * 1000:.2f} ± {std_time * 1000:.2f} ms/seq "
          f"(n={len(data_ids)} real test samples, {N_RUNS_PER_SAMPLE} runs each)")
    print(f"Throughput: {throughput:.1f} fps")

    # 4. Real-time capability check (SingingHead is 30fps)
    print(f"Real-time capable: {'Yes' if throughput >= 30 else 'No'}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Randomly sample IDs from the test set ----
    assert TEST_TXT.exists(), f"test.txt not found: {TEST_TXT}"
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        all_test_ids = [line.strip() for line in f if line.strip()]

    random.seed(RANDOM_SEED)
    sample_ids = random.sample(all_test_ids, min(N_SAMPLES, len(all_test_ids)))

    print(f"Number of samples used for measurement: {len(sample_ids)} / {len(all_test_ids)} "
          f"(random seed={RANDOM_SEED})")
    print(f"Selected samples: {sample_ids}")

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
