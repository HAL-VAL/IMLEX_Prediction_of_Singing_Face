# Prediction of Singing Facial Motions from Musical Features

This repository contains the implementation code for the singing 
facial expression estimation models proposed in this thesis, 
including the Concatenation-based Model and the Cross-Attention-based 
Model.

> **Note:** This repository is currently documented in Japanese. 
> An English translation, along with detailed comments explaining 
> the purpose and usage of each script, is planned for a future 
> update.

## Dataset

This project uses the [SingingHead dataset](https://github.com/wsj-sjtu/SingingHead).

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
