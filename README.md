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

## Contents

- Training scripts
- Inference scripts
- Feature extraction scripts (wav2vec 2.0, MFCC)
- Evaluation scripts

### Citation
```bibtex
@article{wu2023singinghead,
  title={Singinghead: A large-scale 4d dataset for singing head animation},
  author={Wu, Sijing and Li, Yunhao and Zhang, Weitian and Jia, Jun and Zhu, Yucheng and Yan, Yichao and Zhai, Guangtao and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2312.04369},
  year={2023}
}
```
