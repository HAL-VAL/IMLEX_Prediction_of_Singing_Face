import os
import pickle

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# =====================================
# Path
# =====================================

AUDIO_DIR = "../data/SingingHead/bgm_seqs"
FLAME_DIR = "../data/SingingHead/flame_seqs"

# =====================================
# Resize
# =====================================

def resize_to_240(x):

    old_idx = np.linspace(
        0, 1, len(x)
    )

    new_idx = np.linspace(
        0, 1, 240
    )

    return np.interp(
        new_idx,
        old_idx,
        x
    )

# =====================================
# File List
# =====================================

file_list = [
    f for f in os.listdir(FLAME_DIR)
    if f.endswith(".pkl")
]

print(
    f"Total files: {len(file_list)}"
)

# =====================================
# Cache
# =====================================

FEATURE_CACHE = "all_features_9feat_56d_bgm.pkl"

if os.path.exists(FEATURE_CACHE):

    print(f"\nLoading cached features: {FEATURE_CACHE}")

    with open(FEATURE_CACHE, "rb") as f:
        cache = pickle.load(f)

    all_exp = cache["all_exp"]
    music_features = cache["music_features"]

    print("Cache loaded.")

else:

    print("\nCache not found.")
    print("Extracting features...")

    # =====================================
    # Storage
    # =====================================

    all_exp = []

    music_features = {
        "Onset": [],
        "F0": [],
        "Centroid": [],
        "Chroma": [],
        "Contrast": [],
        "Mel": [],
        "MFCC": [],
        "RMS": [],
        "ZCR": []
    }

    # =====================================
    # Main Loop
    # =====================================

    for fname in tqdm(
        file_list,
        desc="Loading"
    ):

        wav_path = os.path.join(
            AUDIO_DIR,
            fname.replace(
                ".pkl",
                "_bgm.wav"
            )
        )

        pkl_path = os.path.join(
            FLAME_DIR,
            fname
        )

        if not os.path.exists(wav_path):
            continue

        # ------------------
        # FLAME
        # ------------------

        with open(
            pkl_path,
            "rb"
        ) as f:

            data = pickle.load(f)

        exp = data["expcodes"]
        pose = data["posecodes"]

        feature = np.concatenate(
            [
                exp,
                pose[:,0:6]
            ],
            axis=1
        )

        # ------------------
        # Audio
        # ------------------

        y, sr = librosa.load(
            wav_path,
            sr=16000
        )

        onset = librosa.onset.onset_strength(
            y=y,
            sr=sr
        )

        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=1000
        )

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )[0]

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr
        )

        chroma = np.mean(
            chroma,
            axis=0
        )

        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr
        )

        contrast = np.mean(
            contrast,
            axis=0
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = np.mean(
            mel,
            axis=0
        )

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        mfcc = np.mean(
            mfcc,
            axis=0
        )

        rms = librosa.feature.rms(
            y=y
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            y=y
        )[0]

        # ------------------
        # Resize
        # ------------------

        onset = resize_to_240(onset)
        f0 = resize_to_240(f0)
        centroid = resize_to_240(centroid)
        chroma = resize_to_240(chroma)
        contrast = resize_to_240(contrast)
        mel = resize_to_240(mel)
        mfcc = resize_to_240(mfcc)
        rms = resize_to_240(rms)

        zcr = resize_to_240(zcr)

        # ------------------
        # Save
        # ------------------

        all_exp.append(feature)

        music_features["Onset"].append(onset)
        music_features["F0"].append(f0)
        music_features["Centroid"].append(centroid)
        music_features["Chroma"].append(chroma)
        music_features["Contrast"].append(contrast)
        music_features["Mel"].append(mel)
        music_features["MFCC"].append(mfcc)
        music_features["RMS"].append(rms)
        music_features["ZCR"].append(zcr)

    # =====================================
    # Cache Save
    # =====================================

    print("\nSaving features...")

    with open(FEATURE_CACHE, "wb") as f:

        pickle.dump(
            {
                "all_exp": all_exp,
                "music_features": music_features
            },
            f
        )

    print(f"Saved: {FEATURE_CACHE}")

# =====================================
# PCA
# =====================================

print(
    "\nPCA..."
)

exp_all = np.concatenate(
    all_exp,
    axis=0
)

scaler = StandardScaler()

exp_all_std = scaler.fit_transform(
    exp_all
)

pca = PCA(
    n_components=10
)

exp_pca = pca.fit_transform(
    exp_all_std
)

print(
    "\nExplained Variance Ratio"
)

for i, r in enumerate(
    pca.explained_variance_ratio_
):

    print(
        f"PC{i+1}: {r:.4f}"
    )

# =====================================
# Music Concatenate
# =====================================

music_concat = {}

for key in music_features:

    music_concat[key] = np.concatenate(
        music_features[key]
    )

# =====================================
# Correlation
# =====================================

corr_matrix = np.zeros(
    (len(music_features),10)
)

pval_matrix = np.zeros(
    (len(music_features),10)
)

music_names = list(
    music_concat.keys()
)

pc_names = [
    f"PC{i+1}"
    for i in range(10)
]

print(
    "\nCorrelation..."
)

for i, feat in enumerate(
    music_names
):

    for j in range(10):

        corr, pval = pearsonr(
            music_concat[feat],
            exp_pca[:,j]
        )

        corr_matrix[i,j] = corr
        pval_matrix[i,j] = pval

# =====================================
# DataFrame
# =====================================

corr_df = pd.DataFrame(
    corr_matrix,
    index=music_names,
    columns=pc_names
)

print(corr_df)

corr_df.to_csv(
    "music_vs_pca_9feat_56d_bgm.csv"
)

# =====================================
# P-value
# =====================================

pval_df = pd.DataFrame(
    pval_matrix,
    index=music_names,
    columns=pc_names
)

print("\nP-values...")
print(pval_df)

pval_df.to_csv(
    "music_vs_pca_pval_9feat_56d_bgm.csv"
)

# 有意な相関のみ表示 (p < 0.05)
sig_mask = pval_matrix < 0.05

print("\nSignificant correlations (p < 0.05):")
sig_corr_df = corr_df.copy()
sig_corr_df[~sig_mask] = float('nan')
print(sig_corr_df.dropna(how='all'))


# =====================================
# Heatmap
# =====================================
# =====================================
# Summary Metrics
# =====================================

print("\n" + "="*50)
print("Summary Metrics")
print("="*50)

# -------------------------------------
# 全体の平均絶対相関
# -------------------------------------

mean_abs_corr = np.mean(
    np.abs(corr_matrix)
)

print(
    f"\nMean Absolute Correlation : {mean_abs_corr:.4f}"
)

# -------------------------------------
# 全体の最大絶対相関
# -------------------------------------

max_abs_corr = np.max(
    np.abs(corr_matrix)
)

max_idx = np.unravel_index(
    np.argmax(np.abs(corr_matrix)),
    corr_matrix.shape
)

print(
    f"Max Absolute Correlation  : {max_abs_corr:.4f}"
)

print(
    f"  Feature : {music_names[max_idx[0]]}"
)

print(
    f"  PC      : {pc_names[max_idx[1]]}"
)

# -------------------------------------
# 寄与率重み付き相関
# -------------------------------------

weighted_corr = {}

for i, feat in enumerate(
    music_names
):

    score = np.sum(
        np.abs(corr_matrix[i]) *
        pca.explained_variance_ratio_
    )

    weighted_corr[feat] = score

print("\nWeighted Correlation")

for feat, score in sorted(
    weighted_corr.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{feat:10s}: {score:.4f}"
    )

# -------------------------------------
# 全体の寄与率重み付き平均
# -------------------------------------

overall_weighted_corr = np.mean(
    list(weighted_corr.values())
)

print(
    f"\nOverall Weighted Correlation : "
    f"{overall_weighted_corr:.4f}"
)

summary_df = pd.DataFrame({
    "Metric": [
        "MeanAbsCorr",
        "MaxAbsCorr",
        "OverallWeightedCorr"
    ],
    "Value": [
        mean_abs_corr,
        max_abs_corr,
        overall_weighted_corr
    ]
})

summary_df.to_csv(
    "summary_metrics_9feat_56d_bgm.csv",
    index=False
)

import os
os.makedirs("pca_figures", exist_ok=True)

plt.figure(
    figsize=(8,5)
)

sns.heatmap(
    corr_df,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-0.5,
    vmax=0.5
)

plt.title(
    "BGM – Combined Parameters (56D)"
)

plt.xlabel("Principal Components")
plt.ylabel("Musical Features")
plt.tight_layout()

plt.savefig(
    os.path.join("pca_figures", "music_vs_pca_heatmap_9feat_56d_bgm.png"),
    dpi=300
)

plt.show()