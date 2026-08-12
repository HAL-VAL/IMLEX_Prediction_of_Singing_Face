import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

def extract_features(input_dir, output_dir, device="cuda"):
    # 1. モデルと抽出器のロード
    print("Loading wav2vec 2.0 model...")
    model_name = "facebook/wav2vec2-base-960h"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    # 2. 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 対象ファイルリストの取得
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} files to process.")

    # 3. 抽出ループ
    for filename in tqdm(wav_files, desc="Extracting wav2vec features"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".wav", ".npy"))
        
        # すでに存在する場合はスキップ（再開時便利）
        if os.path.exists(output_path):
            continue

        try:
            # 音声読み込みとリサンプリング
            wav, sr = torchaudio.load(input_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
            
            # wav2vec形式へ変換
            inputs = feature_extractor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            
            # 特徴量抽出
            with torch.no_grad():
                features = model(**inputs).last_hidden_state.cpu().numpy()
            
            # 保存 (1, seq_len, 768) -> (seq_len, 768) に圧縮
            np.save(output_path, features.squeeze(0))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # スクリプトの場所を起点に、プロジェクトのデータディレクトリを特定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 構造: /src/Visualization/scripts/train_code_voiceandMFCC/xxx.py
    # dataフォルダが /src/Visualization/data にあると仮定してパスを生成
    # 必要に応じて、".." を増やして調整してください
    dataset_base_dir = "/home/nagao2/src/Visualization/data/SingingHead"
    
    # 読み込み元と保存先
    INPUT_DIR = os.path.join(dataset_base_dir, "audio_seqs")
    OUTPUT_DIR = os.path.join(dataset_base_dir, "wav2vec_features")
    
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    extract_features(INPUT_DIR, OUTPUT_DIR)