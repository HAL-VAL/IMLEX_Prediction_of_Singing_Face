"""
FLAME表情推定モデルの評価スクリプト
4モデルを切り替えて評価できる引数対応版（BAスコア対応版）

使い方:
  python evaluation.py --pred_dir predictions_crossattn_volstab --bgm_dir /path/to/bgm_seqs
"""

"""
FLAME表情推定モデルの評価スクリプト
4モデルを切り替えて評価できる引数対応版

使い方:
  python evaluation.py --pred_dir predictions_vol
  python evaluation.py --pred_dir predictions_MFCC
  python evaluation.py --pred_dir predictions_audio_bgm
  python evaluation.py --pred_dir predictions_crossattn
  python evaluation.py --pred_dir predictions_crossattn_pe
  python evaluation.py --pred_dir predictions_crossattn_pe_volstab
  python evaluation.py --pred_dir predictions_crossattn_volstab
  python evaluation.py --pred_dir predictions_nocrossattn-nojaw-seed0
  python evaluation.py --pred_dir predictions_crossattn-jaw-seed0
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
# 次元定義
# ==========================================
EXP_ONLY_DIM    = 50
POSE_NO_JAW_DIM = 6

# ==========================================
# BAスコア計算関数（UniSingerのmetric_3d/metrics.pyのnumpy版）
# 参照元: https://github.com/lisiyao21/Bailando
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
# コマンドライン引数
# ==========================================
parser = argparse.ArgumentParser(description="FLAME表情推定モデルの評価スクリプト")
parser.add_argument("--pred_dir", type=str, required=True,
                    help="推論結果のサブフォルダ名または絶対パス")
parser.add_argument("--output_csv", type=str, default=None,
                    help="CSV出力パス（省略時は pred_dir 名から自動生成）")
parser.add_argument("--dataset_dir", type=str,
                    default=r"D:\MasterDataset\SingingHead\Dataset",
                    help="GT flame_seqs と test.txt があるデータセットディレクトリ")
parser.add_argument("--pred_root", type=str, default=None,
                    help="predictions フォルダの絶対パス")
parser.add_argument("--txt", type=str, default="test.txt",
                    help="評価対象IDリスト（デフォルト: test.txt）")
parser.add_argument("--beat_cache_dir", type=str, default=None,
                    help="precompute_audio_beats.py で作成したビートキャッシュ(.npy)のフォルダ。"
                         "省略時は dataset_dir/beat_cache を使用")
parser.add_argument("--skip_ba", action="store_true",
                    help="BAスコアの計算をスキップする")
args = parser.parse_args()

# ==========================================
# パス設定
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
# test.txt の読み込み
# ==========================================
with open(test_txt_path, "r") as f:
    data_ids = [line.strip().replace(".pkl", "") for line in f if line.strip()]

print(f"モデル: {pred_dir_name}")
print(f"評価対象: {len(data_ids)} 件")
print(f"CSV出力先: {output_csv_path}")


# ==========================================
# 誤差計算ループ
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

        # ---- BA (Beat Align Score)：事前計算済みキャッシュを利用 ----
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
# 集計
# ==========================================
if not results:
    print("評価できたサンプルが0件です。パスを確認してください。")
else:
    keys = [
        "mse_total", "mae_total", "mse_exp", "mse_global", "mse_neck",
        "ppe_global", "ppe_global_neck",
        "mse_vel", "mse_vel_exp", "mse_vel_pose", "jitter", "ba_pose",
    ]

    print(f"\n{'='*60}")
    print(f"評価結果サマリー: {pred_dir_name} ({len(results)} 件）")
    print(f"{'='*60}")
    print(f"{'指標':<20} {'平均':>10} {'最小':>10} {'最大':>10}")
    print(f"{'-'*60}")

    for key in keys:
        vals = [r[key] for r in results if not np.isnan(r[key])]
        if len(vals) == 0:
            print(f"{key:<20} {'N/A':>10}")
            continue
        print(f"{key:<20} {np.mean(vals):>10.6f} {np.min(vals):>10.6f} {np.max(vals):>10.6f}")

    print(f"{'='*60}")

    if errors:
        print(f"\nスキップしたサンプル: {len(errors)} 件")
    if ba_errors:
        print(f"BAスコアを計算できなかったサンプル: {len(ba_errors)} 件（BGM音源が見つからない等）")

    all_keys = ["data_id", "mse_total", "mae_total", "mse_exp", "mae_exp",
                "mse_global", "mae_global", "mse_neck", "mae_neck",
                "ppe_global", "ppe_global_neck", "mse_vel", "mse_vel_exp",
                "mse_vel_pose", "jitter", "ba_pose"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV保存完了: {output_csv_path}")

    print(f"\n--- ワースト10件（mse_total が大きい順）---")
    sorted_results = sorted(results, key=lambda x: x["mse_total"], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1:2d}. {r['data_id']:<25} mse_total={r['mse_total']:.6f}")

    print(f"\n--- ベスト10件（mse_total が小さい順）---")
    for i, r in enumerate(sorted_results[-10:][::-1]):
        print(f"  {i+1:2d}. {r['data_id']:<25} mse_total={r['mse_total']:.6f}")