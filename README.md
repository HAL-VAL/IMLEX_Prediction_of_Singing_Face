# Prediction of Singing Facial Motions from Musical Features

This research was carried out as part of the IMLEX program.

**What is the IMLEX program?** 

[ENGLISH](https://www.imlex.org/) | [JAPANESE](https://imlex.tut.ac.jp/)

This repository contains the implementation code for the singing facial expression estimation models proposed in this master's thesis, titled "Prediction of Singing Facial Motions from Musical Features." It implements two architectures, the Concatenation-based Model and the Cross-Attention-based Model, both of which take a singer's vocal audio and background music (BGM) as input and predict FLAME facial expression and head/neck pose parameters, which can then be rendered into a 3D singing animation.


## Dataset

This study uses the [SingingHead dataset](https://github.com/wsj-sjtu/SingingHead). SingingHead is a large-scale singing dataset that contains singing videos, vocal audio, BGM, and 3D FLAME facial parameters.

Among the data provided in SingingHead, this study uses the following three types of data:

- **Vocal audio**: Singing vocals (`.wav`)
- **BGM audio**: Accompaniment audio (`.wav`)
- **FLAME parameters**: FLAME parameters representing facial motion during singing (`.pkl`)

The FLAME parameters include **facial expressions (expression)** and **head and neck poses (global pose / neck pose)**. 

In this study, these parameters are used as prediction targets, while **jaw pose is excluded from the prediction target**.

## Overview

This repository implements two models for predicting singing facial motions from musical features: a Concatenation-based Model and a Cross-Attention-based Model.

Both models take vocal audio and BGM as inputs and predict FLAME facial parameters consisting of 50 expression parameters and 6 head/neck pose parameters.

### Concatenation-based Model

<img src="readme_pic/architecture_concat.png" width="45%">

The vocal and BGM features are projected into the same feature space and
concatenated before being processed by the Transformer.

### Cross-Attention-based Model

<img src="readme_pic/architecture_crossattn.png" width="45%">

The vocal features are used as queries, while the BGM features are used as keys and values in the cross-attention module. 


Ablation experiments are conducted to investigate the effects of **Positional Encoding (PE)** and the **Volume-based Stability Loss (VolStab)** in the loss function on prediction accuracy.

| Model | Positional Encoding | Volume-based Stability Loss | Training Script |
|---|:---:|:---:|---|
| Concatenation-based | ✓ |  | `scripts/train_code_wav2andMFCC/train.py` |
| Cross-Attention-based |  |  | `scripts/train_code_crossattention/train.py` |
| Cross-Attention-based | ✓ |  | `scripts/train_code_crossattention/train_v2.py` |
| Cross-Attention-based |  | ✓ | `scripts/train_code_crossattention/train_nope.py` |
| Cross-Attention-based | ✓ | ✓| `scripts/train_code_crossattention/train_add_volstab.py` |


## Repository Structure

```
IMLEX_Prediction_of_Singing_Face/
├── FLAME_PyTorch/                 # FLAME model
├── predictions/                   # Directory for storing prediction results
├── rendering_result/              # Rendered videos of prediction results
├── scripts/                       # Scripts for training, inference, and feature extraction
├── calculate_model_performance.py # Utility for measuring model size, number of parameters, and throughput
├── evaluation.py                  # Quantitative evaluation (MSE, MAE, PPE, velocity error, jitter, and Beat Align score)
├── output_mixedwav.py             # Mixes vocal and BGM audio to create audio for rendered videos
├── rendering.py                   # Outputs predicted FLAME sequences as MP4 videos with audio
└── requirements.txt               # Python dependencies
```

## Setup

```bash
git clone --recurse-submodules https://github.com/HAL-VAL/IMLEX_Prediction_of_Singing_Face.git
cd IMLEX_Prediction_of_Singing_Face
pip install -r requirements.txt
```
 
The FLAME model files (`generic_model.pkl`, `flame_static_embedding.pkl`,
`flame_dynamic_embedding.npy`) must be obtained separately from the
official [FLAME project](https://flame.is.tue.mpg.de/) and placed
under the `FLAME_PyTorch` submodule directory.

## Dataset Preparation
 
Download the [SingingHead dataset](https://github.com/wsj-sjtu/SingingHead)
and arrange it under a `data/SingingHead/` directory with, at minimum:
 
```
data/SingingHead/
├── audio_seqs/      # raw vocal .wav files
├── bgm_seqs/        # raw BGM .wav files
├── flame_seqs/      # ground-truth FLAME parameters (.pkl)
├── train.txt        # sample IDs used for training
├── val.txt          # sample IDs used for validation
└── test.txt         # sample IDs used for testing
```

### Feature Extraction
 
Extract vocal features with wav2vec 2.0 and BGM features with MFCC
before training or inference:
 
```bash
python scripts/train_code_wav2andMFCC/wav2vec_extract.py   # -> wav2vec_features/{id}.npy  (T, 768)
python scripts/train_code_wav2andMFCC/extract_mfcc.py      # -> mfcc_features/{id}.npy     (240, 64)
```

## Training
 
Each script trains one model variant and saves only the best
checkpoint (by validation loss) under `checkpoints/`:

```bash
# Concatenation-based model
python scripts/train_code_wav2andMFCC/train.py

# Cross-Attention-based model
python scripts/train_code_crossattention/train.py

# Cross-Attention-based model + Positional Encoding
python scripts/train_code_crossattention/train_v2.py

# Cross-Attention-based model + Volume-based Stability Loss
python scripts/train_code_crossattention/train_nope.py

# Cross-Attention-based model + Positional Encoding + Volume-based Stability Loss
python scripts/train_code_crossattention/train_add_volstab.py
```

Common training parameters are batch_size=64, lr=1e-4, and seq_len=240 (8 seconds at 30 fps). The number of training epochs varies between scripts.

## Inference

Each training script has a matching inference script that loads a
checkpoint and writes predicted FLAME parameters (`shapecode`,
`expcodes`, `posecodes`) to `predictions/<variant_name>/{id}.pkl`.
The jaw pose is not predicted by any model and is copied from the
ground-truth data.

### Inference Scripts

| Model | Positional Encoding | Volume-based Stability Loss | Inference Script |
|---|:---:|:---:|---|
| Concatenation-based | ✓ |  | `scripts/train_code_wav2andMFCC/test.py` | 
| Cross-Attention-based |  |  | `scripts/train_code_crossattention/test.py` | 
| Cross-Attention-based | ✓ |  | `scripts/train_code_crossattention/test_v2.py` | 
| Cross-Attention-based |  | ✓ | `scripts/train_code_crossattention/test_nope.py` | 
| Cross-Attention-based | ✓ | ✓ | `scripts/train_code_crossattention/test_add_volstab.py` | 

All inference scripts use the same command-line arguments. For example:

```bash
# Single sample
python scripts/train_code_crossattention/test.py --id id15_3_1_3

# Batch inference
python scripts/train_code_crossattention/test.py --txt test.txt

# Specify a checkpoint
python scripts/train_code_crossattention/test.py --txt test.txt \
    --checkpoint checkpoints/crossattention_audio_bgm_best_model_ep50.pth
```

To run another model, replace the inference script and checkpoint with the
corresponding ones listed in the table above.

## Evaluation
 
`evaluation.py` compares predictions against ground truth and reports,
per sample and as a summary (mean / min / max), the following metrics:
MSE / MAE of expression and pose, per-frame position error (PPE),
velocity error, jitter (second derivative), and the Beat Align (BA)
score measuring how well predicted head motion aligns with musical
beats.
 
```bash
python evaluation.py --pred_dir predictions_crossattn_pe_volstab
```



### Citation
```bibtex
@article{wu2023singinghead,
  title={Singinghead: A large-scale 4d dataset for singing head animation},
  author={Wu, Sijing and Li, Yunhao and Zhang, Weitian and Jia, Jun and Zhu, Yucheng and Yan, Yichao and Zhai, Guangtao and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2312.04369},
  year={2023}
}
@article{chung2025audio2face,
  title={Audio2face-3d: Audio-driven realistic facial animation for digital avatars},
  author={Chung, Chaeyeon and Fedorov, Ilya and Huang, Michael and Karmanov, Aleksey and Korobchenko, Dmitry and Ribera, Roger and Seol, Yeongho and others},
  journal={arXiv preprint arXiv:2508.16401},
  year={2025}
}
@inproceedings{siyao2022bailando,
  title={Bailando: 3d dance generation by actor-critic gpt with choreographic memory},
  author={Siyao, Li and Yu, Weijiang and Gu, Tianpei and Lin, Chunze and Wang, Quan and Qian, Chen and Loy, Chen Change and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11050--11059},
  year={2022}
}
```
