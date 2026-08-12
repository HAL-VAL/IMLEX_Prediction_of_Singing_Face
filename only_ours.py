import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import numpy as np
import torch
import trimesh
import pyrender
import imageio
from moviepy import ImageSequenceClip, AudioFileClip
from flame_pytorch import FLAME, get_config

for attr in ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str']:
    if not hasattr(np, attr):
        setattr(np, attr, getattr(np, attr + '_', None))
if not hasattr(np, 'str'):
    np.str = str

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# ==========================================
# パス設定
# ==========================================
flame_base_dir   = r"D:\flame"
dataset_base_dir = r"D:\MasterDataset\SingingHead\Dataset"

#pred_dir          = os.path.join(flame_base_dir, "predictions", "predictions_audio_bgm")  # 提案手法のみ predictions_crossattn_pe_volstab
#pred_dir = os.path.join(flame_base_dir, "predictions", "predictions_crossattn_pe_volstab")
pred_dir   = os.path.join(dataset_base_dir, "flame_seqs")
gt_dir   = os.path.join(dataset_base_dir, "flame_seqs")  # 追加
mixed_audio_dir   = os.path.join(flame_base_dir, "mixed_audio")   # ミックス済み音声フォルダ
output_video_dir  = os.path.join(flame_base_dir, "ours_only_videos")
temp_img_dir      = os.path.join(flame_base_dir, "temp_rendered_frames")
os.makedirs(output_video_dir, exist_ok=True)

# ==========================================
# 出力したいIDをここで指定
# ==========================================
#target_ids = ["id22_15_0_7", "id57_2_0_9", "id21_5_0_11", "id49_14_0_19"]  # ← 好きなIDを列挙
target_ids = ["id0_10_0_26"]

# ==========================================
# FLAMEモデルのロード
# ==========================================
print("FLAMEモデルをロード中...")
config = get_config()
config.batch_size = 1
config.flame_model_path                = r"D:\flame\FLAME_PyTorch\model\generic_model.pkl"
config.static_landmark_embedding_path  = r"D:\flame\FLAME_PyTorch\model\flame_static_embedding.pkl"
config.dynamic_landmark_embedding_path = r"D:\flame\FLAME_PyTorch\model\flame_dynamic_embedding.npy"

flamelayer = FLAME(config).to(device)
faces = flamelayer.faces

renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)

def render_sequence(shape_params, expressions, poses, name_prefix, num_frames):
    os.makedirs(temp_img_dir, exist_ok=True)
    saved_files = []

    camera_params = {'c': np.array([400, 400]), 'f': np.array([2377.0, 2377.0])}
    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND', baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8, roughnessFactor=0.8
    )

    for frame_idx in range(num_frames):
        current_exp = expressions[frame_idx:frame_idx + 1]
        global_jaw = torch.cat([
            poses[frame_idx:frame_idx + 1, :3],
            poses[frame_idx:frame_idx + 1, 6:9],
        ], dim=1)
        neck_pose = poses[frame_idx:frame_idx + 1, 3:6]

        vertices, _ = flamelayer(shape_params, current_exp, global_jaw, neck_pose=neck_pose)
        vertices = vertices[0].detach().cpu().numpy()
        vertices = vertices - vertices.mean(axis=0)

        tri_mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

        #scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0, 1.0])
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[1.0, 1.0, 1.0, 1.0]) # 白
        scene.add(mesh, pose=np.eye(4))

        camera = pyrender.IntrinsicsCamera(
            fx=camera_params['f'][0], fy=camera_params['f'][1],
            cx=camera_params['c'][0], cy=camera_params['c'][1],
            znear=0.01, zfar=3.0
        )
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0, -0.03, 1.0]
        scene.add(camera, pose=cam_pose)

        angle = np.pi / 6.0
        light = pyrender.DirectionalLight(color=np.array([1., 1., 1.]), intensity=2.0)
        pos = cam_pose[:3, 3]
        for rot_vec in [np.zeros(3), [angle,0,0], [-angle,0,0], [0,-angle,0], [0,angle,0]]:
            import cv2
            light_pose = np.eye(4)
            light_pose[:3, 3] = cv2.Rodrigues(np.array(rot_vec, dtype=np.float64))[0].dot(pos)
            scene.add(light, pose=light_pose.copy())

        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        color, _ = renderer.render(scene, flags=flags)

        file_path = os.path.join(temp_img_dir, f"{name_prefix}_{frame_idx:05d}.png")
        imageio.imwrite(file_path, color)
        saved_files.append(file_path)
    return saved_files

# ==========================================
# メインループ：指定IDのみ処理
# ==========================================
print(f"【処理開始】対象ID {len(target_ids)}件")
print(f"出力先: {output_video_dir}")

for index, data_id in enumerate(target_ids):
    print(f"\n[{index + 1}/{len(target_ids)}] 処理中: {data_id}")

    # 変更後
    pred_path = os.path.join(pred_dir, f"{data_id}.pkl")
    gt_path   = os.path.join(gt_dir, f"{data_id}.pkl")

    if not os.path.exists(pred_path):
        print(f"  スキップ: 予測pklが見つかりません: {pred_path}")
        continue
    if not os.path.exists(gt_path):
        print(f"  スキップ: GT pklが見つかりません: {gt_path}")
        continue

    # 予測（exp・poseはこちらを使用）
    with open(pred_path, 'rb') as f:
        pd = pickle.load(f, encoding='latin1')

    # GT（shapeはこちらを使用）
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f, encoding='latin1')

    shape_np     = np.array(gt['shapecode']).reshape(1, -1)[:, :300]   # ← GTから取得
    shape_params = torch.tensor(shape_np, dtype=torch.float32).to(device)
    exp  = torch.tensor(pd['expcodes'],  dtype=torch.float32).to(device)  # ← predictionのまま
    pose = torch.tensor(pd['posecodes'], dtype=torch.float32).to(device)  # ← predictionのまま

    num_frames = len(exp)
    print(f"  フレーム数: {num_frames}")

    print("  レンダリング中...")
    images = render_sequence(shape_params, exp, pose, f"GT_{data_id}", num_frames)

    fps = 30
    clip = ImageSequenceClip(images, fps=fps)

    # ---- 音声（ミックス済みファイルをそのまま使用） ----
    mixed_path = None
    for candidate_name in [f"{data_id}_mixed", data_id]:
        for ext in [".wav", ".mp3"]:
            candidate = os.path.join(mixed_audio_dir, f"{candidate_name}{ext}")
            if os.path.exists(candidate):
                mixed_path = candidate
                break
        if mixed_path:
            break

    if mixed_path is not None:
        print(f"  音声: {mixed_path}")
        mixed_audio = AudioFileClip(mixed_path)
        if mixed_audio.duration > clip.duration:
            mixed_audio = mixed_audio.subclipped(0, clip.duration)
        clip = clip.with_audio(mixed_audio)
    else:
        print(f"  警告: ミックス音声が見つかりません ({data_id})")

    out_path = os.path.join(output_video_dir, f"GT_{data_id}.mp4")
    print(f"  出力: {out_path}")
    clip.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", logger=None)

    for img in images:
        if os.path.exists(img):
            os.remove(img)

renderer.delete()
print(f"\n完了！ 動画保存先: {output_video_dir}")