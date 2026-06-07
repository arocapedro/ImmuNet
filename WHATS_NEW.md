# What's New — High-Level Summary

This repository is a **modernized PyTorch fork** of the original [ImmuNet](https://github.com/jtextor/immunet) codebase, which was implemented in TensorFlow 1.14 / Keras 2.3 on Python 3.6. The science (segmentation-free immune-cell detection and phenotyping in multiplex IHC images) is unchanged — the engineering around it has been rebuilt.

## In one paragraph

The model and training pipeline have been ported from TensorFlow 1.14 + Keras 2.3 to **PyTorch 2.5 on Python 3.12**. Two additional backbones (`UImmNet`, `DAPImmuNet`) sit alongside a faithful PyTorch reimplementation of the original architecture, all sharing a common `BaseImmunet` interface. The training loop has been rewritten with TensorBoard logging, mixed-precision (AMP), early stopping, configurable losses and weighting, branch/backbone freezing, and CUDA / CPU / Apple-Silicon (MPS) device support. A modern `torchvision.transforms.v2` augmentation stack and multi-process sample preparation replace the old Keras `ImageDataGenerator`. New modules (`prediction.py`, `visualization.py`, `train_utils.py`, `models_parts.py`) split out responsibilities that were previously tangled together. A pretrained PyTorch checkpoint (`immunet_zenodo.pth`) ships in the repo, and a Keras-→-PyTorch weight-conversion utility lets the original Zenodo model be imported without retraining.

## What changed at a glance

| Area | Original | This fork |
|---|---|---|
| Framework | TensorFlow 1.14, Keras 2.3 | PyTorch 2.5.1, torchvision 0.20.1 |
| Python | 3.6.9 | 3.12.8 |
| Model file | `immunet.h5` (Keras HDF5) | `immunet.pth` (PyTorch state dict) |
| Backbones | 1 (original) | 3 (`ImmuNet`, `UImmNet`, `DAPImmuNet`) |
| Training loop | `model.fit(...)` | Explicit PyTorch loop with AMP, early stopping, TensorBoard |
| Augmentations | `ImageDataGenerator` | `torchvision.transforms.v2` pipeline |
| Devices | CUDA / CPU | CUDA / CPU / **MPS (Apple Silicon)** |
| Pretrained weights | Download `.h5` from Zenodo | `immunet_zenodo.pth` ships in repo |
| New modules | — | `models_parts.py`, `train_utils.py`, `prediction.py`, `visualization.py` |
| Type hints | minimal | throughout model / training / prediction code |

## Why this fork exists

The upstream codebase pins TensorFlow 1.14 (last released in 2019) and Python 3.6 (EOL 2021), which makes it increasingly painful to install on current systems and incompatible with modern GPUs and Apple Silicon. This fork removes that constraint and modernizes the engineering surface so the model can be trained, fine-tuned, and extended with current tooling — without changing the published architecture or the science behind it.

For the full per-area diff, see [`CHANGELOG.md`](CHANGELOG.md).
