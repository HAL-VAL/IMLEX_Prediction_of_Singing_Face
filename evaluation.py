"""
FLAME Expression Estimation Model Evaluation Script
Argument-driven version that can switch between 4 models (with BA score support)

Usage:
  python evaluation.py --pred_dir predictions_crossattn_volstab --bgm_dir /path/to/bgm_seqs
"""

"""
FLAME Expression Estimation Model Evaluation Script

Usage:
  python evaluation.py --pred_dir predictions_audio_bgm
  python evaluation.py --pred_dir predictions_crossattn
  python evaluation.py --pred_dir predictions_crossattn_pe
  python evaluation.py --pred_dir predictions_crossattn_pe_volstab
  python evaluation.py --pred_dir predictions_crossattn_volstab

"""

import os
import csv
import argparse
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from tqdm import tqdm

# ==========================================
# Dimension definitions
# ==========================================
EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6

# ==========================================
# BA score calculation functions (numpy port of UniSinger's metric_3d/metrics.py)
# Source: https://github.com/lisiyao21/Bailando
# ==========================================


def calc_db(motion_seq):
    """
    motion_seq: (nframe, 6)  -- global(3) + neck(3)
    """
    seq = np.array(motion_seq)
    velocity = np.sqrt(np.sum((seq[1:] - seq[:-1]) ** 2, axis=-1))
    velocity = gaussian_filter(velocity, 5)
    motion_beats = argrelextrema(velocity, np.less)
    return motion_beats, len(velocity)


def BA(music_beats, motion_beats):
    if len(music_beats) == 0:
        music_beats = np.array([0], dtype=np.int64)
    if len(motion_beats[0]) == 0:
        return 0.0
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)




# ==========================================
# Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="FLAME Expression Estimation Model Evaluation Script")
parser.add_argument("--pred_dir", type=str, required=True,
                    help="Subfolder name or absolute path of the prediction results")
parser.add_argument("--output_csv", type=str, default=None,
                    help="CSV output path (auto-generated from pred_dir name if omitted)")
parser.add_argument("--dataset_dir", type=str,
                    default=r"D:\MasterDataset\SingingHead\Dataset",
                    help="Dataset directory containing GT flame_seqs and test.txt")
parser.add_argument("--pred_root", type=str, default=None,
                    help="Absolute path of the predictions folder")
parser.add_argument("--txt", type=str, default="test.txt",
                    help="ID list to evaluate (default: test.txt)")
parser.add_argument("--beat_cache_dir", type=str, default=None,
                    help="Folder containing the beat cache (.npy) created by precompute_audio_beats.py. "
                         "If omitted, dataset_dir/beat_cache is used")
parser.add_argument("--skip_ba", action="store_true",
                    help="Skip computing the BA score")
args = parser.parse_args()

# ==========================================
# Path configuration
# ==========================================
dataset_base_dir = args.dataset_dir
gt_flame_dir     = os.path.join(dataset_base_dir, "flame_seqs")

script_dir = os.path.dirname(os.path.abspath(__file__))
pred_root  = args.pred_root or os.path.join(script_dir, "predictions")

beat_cache_dir = args.beat_cache_dir or os.path.join(dataset_base_dir, "beat_cache")

if os.path.isabs(args.pred_dir):
    pred_dir = args.pred_dir
else:
    pred_dir = os.path.join(pred_root, args.pred_dir)

pred_dir_name   = os.path.basename(pred_dir.rstrip("/\\"))
output_csv_path = args.output_csv or os.path.join(
    pred_root, f"evaluation_{pred_dir_name}.csv"
)

test_txt_path = args.txt if os.path.isabs(args.txt) else os.path.join(dataset_base_dir, args.txt)

# ==========================================
# Read test.txt
# ==========================================
with open(test_txt_path, "r") as f:
    data_ids = [line.strip().replace(".pkl", "") for line in f if line.strip()]

print(f"Model: {pred_dir_name}")
print(f"Evaluation targets: {len(data_ids)} samples")
print(f"CSV output path: {output_csv_path}")


# ==========================================
# Error computation loop
# ==========================================
results = []
errors  = []
ba_errors = []

for data_id in tqdm(data_ids, desc="Evaluating"):
    gt_path   = os.path.join(gt_flame_dir, f"{data_id}.pkl")
    pred_path = os.path.join(pred_dir,     f"{data_id}.pkl")

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        errors.append(data_id)
        continue

    try:
        with open(gt_path, "rb") as f:
            gt_data = pickle.load(f, encoding="latin1")
        with open(pred_path, "rb") as f:
            pred_data = pickle.load(f, encoding="latin1")

        gt_exp   = np.array(gt_data["expcodes"],  dtype=np.float32)
        gt_pose  = np.array(gt_data["posecodes"], dtype=np.float32)
        pred_exp  = np.array(pred_data["expcodes"],  dtype=np.float32)
        pred_pose = np.array(pred_data["posecodes"], dtype=np.float32)

        n = min(len(gt_exp), len(pred_exp))
        gt_exp,  pred_exp  = gt_exp[:n],  pred_exp[:n]
        gt_pose, pred_pose = gt_pose[:n], pred_pose[:n]

        mse_exp = np.mean((gt_exp - pred_exp) ** 2)
        mae_exp = np.mean(np.abs(gt_exp - pred_exp))

        mse_global = np.mean((gt_pose[:, :3] - pred_pose[:, :3]) ** 2)
        mae_global = np.mean(np.abs(gt_pose[:, :3] - pred_pose[:, :3]))

        mse_neck = np.mean((gt_pose[:, 3:6] - pred_pose[:, 3:6]) ** 2)
        mae_neck = np.mean(np.abs(gt_pose[:, 3:6] - pred_pose[:, 3:6]))

        ppe_global = np.mean(np.linalg.norm(
            gt_pose[:, :3] - pred_pose[:, :3], axis=-1
        ))
        ppe_global_neck = np.mean(np.linalg.norm(
            gt_pose[:, :6] - pred_pose[:, :6], axis=-1
        ))

        gt_all   = np.concatenate([gt_exp,   gt_pose[:, :6]], axis=-1)
        pred_all = np.concatenate([pred_exp, pred_pose[:, :6]], axis=-1)
        mse_total = np.mean((gt_all - pred_all) ** 2)
        mae_total = np.mean(np.abs(gt_all - pred_all))

        gt_vel   = np.diff(gt_all,   axis=0)
        pred_vel = np.diff(pred_all, axis=0)
        mse_vel  = np.mean((gt_vel - pred_vel) ** 2)

        gt_vel_exp   = np.diff(gt_exp,   axis=0)
        pred_vel_exp = np.diff(pred_exp, axis=0)
        mse_vel_exp  = np.mean((gt_vel_exp - pred_vel_exp) ** 2)

        gt_vel_pose   = np.diff(gt_pose[:, :6], axis=0)
        pred_vel_pose = np.diff(pred_pose[:, :6], axis=0)
        mse_vel_pose  = np.mean((gt_vel_pose - pred_vel_pose) ** 2)

        pred_accel = np.diff(pred_all, n=2, axis=0)
        jitter = np.mean(pred_accel ** 2)

        # ---- BA (Beat Align Score): uses a precomputed cache ----
        ba_pose = np.nan
        if not args.skip_ba:
            beat_cache_path = os.path.join(beat_cache_dir, f"{data_id}.npy")
            if os.path.exists(beat_cache_path):
                try:
                    beats_one_hot = np.load(beat_cache_path)
                    motion_beats, length = calc_db(pred_pose[:, :6])
                    beats = beats_one_hot[:length].astype(bool)
                    audio_beats = np.arange(len(beats))[beats]
                    ba_pose = BA(audio_beats, motion_beats)
                except Exception as e:
                    ba_errors.append(data_id)
            else:
                ba_errors.append(data_id)

        results.append({
            "data_id":         data_id,
            "mse_total":       mse_total,
            "mae_total":       mae_total,
            "mse_exp":         mse_exp,
            "mae_exp":         mae_exp,
            "mse_global":      mse_global,
            "mae_global":      mae_global,
            "mse_neck":        mse_neck,
            "mae_neck":        mae_neck,
            "ppe_global":      ppe_global,
            "ppe_global_neck": ppe_global_neck,
            "mse_vel":         mse_vel,
            "mse_vel_exp":     mse_vel_exp,
            "mse_vel_pose":    mse_vel_pose,
            "jitter":          jitter,
            "ba_pose":         ba_pose,
        })

    except Exception as e:
        errors.append(data_id)
        print(f"\nERROR: {data_id}: {e}")

# ==========================================
# Aggregation
# ==========================================
if not results:
    print("0 samples were successfully evaluated. Please check the paths.")
else:
    keys = [
        "mse_total", "mae_total", "mse_exp", "mse_global", "mse_neck",
        "ppe_global", "ppe_global_neck",
        "mse_vel", "mse_vel_exp", "mse_vel_pose", "jitter", "ba_pose",
    ]

    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {pred_dir_name} ({len(results)} samples)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*60}")

    for key in keys:
        vals = [r[key] for r in results if not np.isnan(r[key])]
        if len(vals) == 0:
            print(f"{key:<20} {'N/A':>10}")
            continue
        print(f"{key:<20} {np.mean(vals):>10.6f} {np.min(vals):>10.6f} {np.max(vals):>10.6f}")

    print(f"{'='*60}")

    if errors:
        print(f"\nSkipped samples: {len(errors)}")
    if ba_errors:
        print(f"Samples for which the BA score could not be computed: {len(ba_errors)} (e.g. BGM audio source not found)")

    all_keys = ["data_id", "mse_total", "mae_total", "mse_exp", "mae_exp",
                "mse_global", "mae_global", "mse_neck", "mae_neck",
                "ppe_global", "ppe_global_neck", "mse_vel", "mse_vel_exp",
                "mse_vel_pose", "jitter", "ba_pose"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV saved: {output_csv_path}")

    print(f"\n--- Worst 10 (sorted by descending mse_total) ---")
    sorted_results = sorted(results, key=lambda x: x["mse_total"], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1:2d}. {r['data_id']:<25} mse_total={r['mse_total']:.6f}")

    print(f"\n--- Best 10 (sorted by ascending mse_total) ---")
    for i, r in enumerate(sorted_results[-10:][::-1]):
        print(f"  {i+1:2d}. {r['data_id']:<25} mse_total={r['mse_total']:.6f}")
