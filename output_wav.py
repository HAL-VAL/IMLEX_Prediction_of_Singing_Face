import os
import numpy as np
from typing import Optional
from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx import MultiplyVolume

# ==========================================
# 1. パス設定（元コードと同じ構成）
# ==========================================
flame_base_dir    = r"D:\flame"
dataset_base_dir  = r"D:\MasterDataset\SingingHead\Dataset"

bgm_folder   = os.path.join(dataset_base_dir, "bgm_seqs")
vocal_folder = os.path.join(dataset_base_dir, "audio_seqs")   # 歌唱音声フォルダ

output_audio_dir = os.path.join(flame_base_dir, "mixed_audio")
os.makedirs(output_audio_dir, exist_ok=True)

# ==========================================
# 2. 音量設定
#    単純な倍率ではなく「ピーク音量を目標dBFSに正規化」する方式。
#    元の音声がどれだけ小さくても指定した音量まで持ち上がる。
# ==========================================
VOCAL_TARGET_DBFS = -3.0   # Vocalのピークをこの音量(dBFS)に揃える
BGM_TARGET_DBFS    = -12.0  # BGMのピークをこの音量(dBFS)に揃える（Vocalより下げておく）

# ==========================================
# 3. 対象IDの指定
#    ここに合成したいdata_idを列挙する
# ==========================================
target_ids = [
    "id22_15_0_7",
    "id57_2_0_9",
    "id21_5_0_11",
    "id5_31_0_21",
    "id5_32_0_16",
    "id49_14_0_19",
    "id0_10_0_26"
    # 必要に応じて追加
]


def peak_dbfs(clip) -> float:
    """クリップのピーク音量をdBFSで返す（無音なら-infを返す）"""
    arr = clip.to_soundarray()
    peak = np.max(np.abs(arr))
    if peak <= 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def normalize_to_dbfs(clip, target_dbfs: float):
    """クリップのピーク音量が target_dbfs になるよう倍率をかけて返す"""
    current = peak_dbfs(clip)
    if current == -np.inf:
        print("    警告: 無音のため正規化をスキップします")
        return clip
    gain_db = target_dbfs - current
    factor = 10.0 ** (gain_db / 20.0)
    print(f"    現在のピーク: {current:.1f} dBFS → 目標: {target_dbfs:.1f} dBFS (倍率 x{factor:.2f})")
    return clip.with_effects([MultiplyVolume(factor)])


def mix_bgm_and_vocal(data_id: str) -> Optional[str]:
    """指定したdata_idのBGMとVocalを合成し、1つの音声ファイルとして書き出す。
    成功した場合は出力パスを返す。どちらも見つからない場合はNoneを返す。
    """
    audio_clips_to_mix = []

    # ---- BGM ----
    bgm_clip = None
    for ext in [".wav", ".mp3"]:
        bgm_path = os.path.join(bgm_folder, f"{data_id}_bgm{ext}")
        if os.path.exists(bgm_path):
            print(f"  BGM: {bgm_path}")
            bgm_clip = AudioFileClip(bgm_path)
            bgm_clip = normalize_to_dbfs(bgm_clip, BGM_TARGET_DBFS)
            audio_clips_to_mix.append(bgm_clip)
            break
    if bgm_clip is None:
        print(f"  警告: BGMが見つかりません ({data_id}_bgm.[wav/mp3])")

    # ---- Vocal ----
    vocal_clip = None
    for ext in [".wav", ".mp3"]:
        vocal_path = os.path.join(vocal_folder, f"{data_id}{ext}")
        if os.path.exists(vocal_path):
            print(f"  Vocal: {vocal_path}")
            vocal_clip = AudioFileClip(vocal_path)
            vocal_clip = normalize_to_dbfs(vocal_clip, VOCAL_TARGET_DBFS)
            audio_clips_to_mix.append(vocal_clip)
            break
    if vocal_clip is None:
        print(f"  警告: Vocalが見つかりません ({data_id}.[wav/mp3])")

    if not audio_clips_to_mix:
        print(f"  スキップ: {data_id} の音声が1つも見つかりませんでした")
        return None

    # ---- ミックス ----
    mixed_audio = CompositeAudioClip(audio_clips_to_mix)

    # ---- 書き出し ----
    out_path = os.path.join(output_audio_dir, f"{data_id}_mixed.wav")
    print(f"  出力: {out_path}")
    mixed_audio.write_audiofile(out_path, logger=None)

    # ---- クリップを閉じてリソース解放 ----
    for clip in audio_clips_to_mix:
        clip.close()
    mixed_audio.close()

    return out_path


if __name__ == "__main__":
    print(f"【処理開始】対象ID数: {len(target_ids)}")
    print(f"出力先: {output_audio_dir}\n")

    for index, data_id in enumerate(target_ids):
        print(f"[{index + 1}/{len(target_ids)}] 処理中: {data_id}")
        mix_bgm_and_vocal(data_id)
        print()

    print("完了！")