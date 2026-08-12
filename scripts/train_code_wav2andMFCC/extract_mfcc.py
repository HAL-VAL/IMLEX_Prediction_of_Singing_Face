import os
import librosa
import numpy as np
from tqdm import tqdm

# ==========================================
# Config
# ==========================================

DATASET_DIR = "/home/nagao2/src/Visualization/data/SingingHead"
BGM_DIR = os.path.join(DATASET_DIR, "bgm_seqs")
OUTPUT_DIR = os.path.join(DATASET_DIR, "mfcc_features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_MFCC = 64
SEQ_LEN = 240

# ==========================================
# 補間関数
# ==========================================

def resize_feature(feat, target_len=240):
    """
    feat: (T, D)
    return: (240, D)
    """

    old_len = feat.shape[0]

    if old_len == target_len:
        return feat

    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, target_len)

    out = np.zeros((target_len, feat.shape[1]), dtype=np.float32)

    for d in range(feat.shape[1]):
        out[:, d] = np.interp(
            new_idx,
            old_idx,
            feat[:, d]
        )

    return out

# ==========================================
# ファイル一覧
# ==========================================

wav_files = sorted(
    f for f in os.listdir(BGM_DIR)
    if f.endswith(".wav")
)

print(f"Found {len(wav_files)} files")

# ==========================================
# MFCC抽出
# ==========================================

for wav_name in tqdm(wav_files, desc="Extract MFCC"):

    input_path = os.path.join(BGM_DIR, wav_name)

    save_name = wav_name.replace(
        "_bgm.wav",
        ".npy"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        save_name
    )

    if os.path.exists(output_path):
        continue

    try:
        y, sr = librosa.load(
            input_path,
            sr=None
        )

        hop_length = int(sr / 30)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            hop_length=hop_length
        )

        # (64, T) → (T, 64)
        mfcc = mfcc.T

        # 240フレームへ補間
        mfcc = resize_feature(
            mfcc,
            SEQ_LEN
        )

        np.save(
            output_path,
            mfcc.astype(np.float32)
        )

    except Exception as e:
        print(f"ERROR: {wav_name}")
        print(e)

print()
print("Saved to:")
print(OUTPUT_DIR)