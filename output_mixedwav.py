import os
import numpy as np
from typing import Optional
from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx import MultiplyVolume

# ==========================================
# 1. Path configuration
# ==========================================
flame_base_dir    = r"D:\flame"
dataset_base_dir  = r"D:\MasterDataset\SingingHead\Dataset"

bgm_folder   = os.path.join(dataset_base_dir, "bgm_seqs")
vocal_folder = os.path.join(dataset_base_dir, "audio_seqs")   # Folder containing the singing vocal audio

output_audio_dir = os.path.join(flame_base_dir, "mixed_audio")
os.makedirs(output_audio_dir, exist_ok=True)

# ==========================================
# 2. Volume settings
#    Uses a "normalize peak volume to a target dBFS" approach.
#    No matter how quiet the original audio is, it gets boosted up
#    to the specified volume.
# ==========================================
VOCAL_TARGET_DBFS = -3.0   # Align the vocal's peak to this volume (dBFS)
BGM_TARGET_DBFS    = -12.0  # Align the BGM's peak to this volume (dBFS) (kept lower than vocal)

# ==========================================
# 3. Target ID specification
#    List the data_ids you want to mix here
# ==========================================
target_ids = [
    "id22_15_0_7",
    "id57_2_0_9",
    "id21_5_0_11",
    "id5_31_0_21",
    "id5_32_0_16",
    "id49_14_0_19",
    "id0_10_0_26",
    "id1_18_1_5",
    "id_16_19_0_17"
    # Add more as needed
]


def peak_dbfs(clip) -> float:
    """Return the clip's peak volume in dBFS (returns -inf if silent)"""
    arr = clip.to_soundarray()
    peak = np.max(np.abs(arr))
    if peak <= 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def normalize_to_dbfs(clip, target_dbfs: float):
    """Return the clip scaled so its peak volume becomes target_dbfs"""
    current = peak_dbfs(clip)
    if current == -np.inf:
        print("    Warning: skipping normalization because the audio is silent")
        return clip
    gain_db = target_dbfs - current
    factor = 10.0 ** (gain_db / 20.0)
    print(f"    Current peak: {current:.1f} dBFS -> Target: {target_dbfs:.1f} dBFS (factor x{factor:.2f})")
    return clip.with_effects([MultiplyVolume(factor)])


def mix_bgm_and_vocal(data_id: str) -> Optional[str]:
    """Mix the BGM and vocal for the given data_id and write out a single audio file.
    Returns the output path on success. Returns None if neither file is found.
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
        print(f"  Warning: BGM not found ({data_id}_bgm.[wav/mp3])")

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
        print(f"  Warning: Vocal not found ({data_id}.[wav/mp3])")

    if not audio_clips_to_mix:
        print(f"  Skipping: no audio found at all for {data_id}")
        return None

    # ---- Mix ----
    mixed_audio = CompositeAudioClip(audio_clips_to_mix)

    # ---- Write out ----
    out_path = os.path.join(output_audio_dir, f"{data_id}_mixed.wav")
    print(f"  Output: {out_path}")
    mixed_audio.write_audiofile(out_path, logger=None)

    # ---- Close clips to release resources ----
    for clip in audio_clips_to_mix:
        clip.close()
    mixed_audio.close()

    return out_path


if __name__ == "__main__":
    print(f"[Starting processing] Number of target IDs: {len(target_ids)}")
    print(f"Output directory: {output_audio_dir}\n")

    for index, data_id in enumerate(target_ids):
        print(f"[{index + 1}/{len(target_ids)}] Processing: {data_id}")
        mix_bgm_and_vocal(data_id)
        print()

    print("Done!")
