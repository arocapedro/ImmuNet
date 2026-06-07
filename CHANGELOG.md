# Changelog

All notable changes to this fork of ImmuNet are documented here. This fork is a substantial modernization of the upstream `jtextor/immunet` repository, which implemented the model in TensorFlow 1.14 / Keras 2.3 on Python 3.6.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project loosely follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — Modernized PyTorch port

This release is a ground-up rewrite of the model and training pipeline on top of PyTorch 2.5. It is **not** drop-in compatible with the original repository: model files, command-line interfaces, and several module boundaries have changed. A weight-conversion utility is provided to migrate the published Keras checkpoint to PyTorch.

### Major changes

#### Framework migration: TensorFlow/Keras → PyTorch
- Replaced TensorFlow 1.14 + Keras 2.3 with **PyTorch 2.5.1 + torchvision 0.20.1**.
- Removed dependencies on `tensorflow-gpu`, `keras`, `h5py`, `bson`, and the legacy `csbdeep==0.6.0` pin.
- Migrated the Python runtime from **3.6.9 → 3.12.8**.
- Migrated NumPy to the **2.x series** (`numpy==2.2.0`); aligned `scipy`, `pandas`, `scikit-image`, `scikit-learn`, `pillow`, and `tifffile` to current versions.
- Model checkpoint format changed from **`.h5` (Keras HDF5) → `.pth` (PyTorch state dict)**.

#### Model architecture
- Introduced an abstract `BaseImmunet` (`nn.Module`) with shared utilities for reflection-padded inference, gradient/branch freezing, and weight loading from disk.
- Reimplemented the original architecture as `ImmuNet`, decomposed into reusable `nn.Module` building blocks (`ImmuNetMainBlock`, `ImmuNetSkipBlock`, `ImmuNetEndBlock`, `ImmuNetOutputBranch`) in the new `immunet/models_parts.py` module.
- Added two **new backbones** alongside the original:
  - `UImmNet` — a U-Net-style encoder/decoder backbone for denser feature maps.
  - `DAPImmuNet` — a single-channel (DAPI-only) variant that predicts cell presence without phenotype outputs.
- Added a `BaseImmunet.from_path(...)` classmethod that auto-detects input channels and marker counts from the saved state dict and instantiates the correct backbone.
- Added `ImmuNet.from_keras_weights(...)` to convert weights produced by the original Keras model into the PyTorch model (handles transpose conventions for `Conv2D` and `Dense` layers, as well as `BatchNorm` running statistics).

#### Training pipeline rewrite (`immunet/train.py`, `immunet/train_utils.py`)
- Replaced the Keras `model.fit(...)` loop with an explicit PyTorch training loop and a structured `TrainConfig` dataclass.
- Added **TensorBoard** logging via `torch.utils.tensorboard.SummaryWriter` (training/validation losses, optional model graph).
- Added **mixed-precision training** with `torch.amp.autocast` and `GradScaler` (`--autocast on/off`).
- Added **early stopping** (`EarlyStopper`) configurable via `--early_stopper_patience` and `--early_stopper_delta`.
- Added selectable **loss functions** (`--loss_function MSE|BCE`) and **per-channel loss weighting** (`--weighted_loss`, `--positive_weights`).
- Added **branch/backbone freezing** flags (`--freeze_backbone`, `--freeze_distance`, `--freeze_phenotype`) to support fine-tuning workflows.
- Added **device selection** (`--device_type cpu|cuda|mps`, `--device_id`) — Apple Silicon (`mps`) is now a first-class option.
- Added structured argument groups (General / Data / Model / Training / Early Stopping / Advanced) and persistent run logging via the `logging` module.
- Run output is now namespaced as `<output_folder>/<YYYYMMDD-HHMMSS>_<experiment_keyword>/` so multiple experiments coexist cleanly.

#### Data pipeline rewrite (`immunet/data_processing.py`)
- Replaced the Keras `ImageDataGenerator` with a PyTorch `Dataset` + `DataLoader` pipeline (`CustomImageDataset`).
- Added a **modern augmentation stack** built on `torchvision.transforms.v2`:
  - `RandomTransformations` (rotations, flips), `RandomBrightness`, `GaussianNoise`, `RandomBlackoutChannels`, `RandomCrop`, and a `ToNHWC` adapter.
- Added **multiprocess sample preparation** via the `multiprocess` package (`--single_thread` to opt out).
- Added support for runtime channel removal during training (`--remove_channel`) to ablate input modalities, and for overriding the default background-type set (`--background_types`).

#### New modules
- `immunet/models_parts.py` — reusable PyTorch building blocks (`DoubleConv`, `Down`, `Up`, `OutConv`, `ImmuNetMainBlock`, `ImmuNetSkipBlock`, `ImmuNetEndBlock`, `ImmuNetOutputBranch`, `ConvSigmoid`).
- `immunet/train_utils.py` — `TrainConfig`, `EarlyStopper`, `EnhancedJSONEncoder`, `HistoryVis`, training-loop helpers, AMP setup, model/loss/optimizer factories.
- `immunet/prediction.py` — `ImmuNetPredictionHandler` class encapsulating model loading, prediction, blob detection, and matching against annotations (replaces ad hoc inference helpers).
- `immunet/visualization.py` — confusion matrices, error bars, KDE / AUC plots, Clopper–Pearson intervals; consumed by `evaluation.py`.

#### Configuration
- `immunet/config.py` now points to `immunet.pth` (was `immunet.h5`) and removes the legacy suffixed-prediction-file constant. Existing path conventions (`data/tilecache`, `data/annotations`, `train_output`, etc.) are preserved.

#### Pretrained weights
- Added the converted Zenodo checkpoint **`immunet_zenodo.pth`** at the repository root for direct PyTorch use; downloading the legacy `immunet.h5` from Zenodo is no longer required for inference.

### Tooling

- `environment.yml` upgraded to Python 3.12.8 with PyTorch 2.5.1, modern numpy/scipy/pandas/scikit-* stack, and dev tools (`black`, `pre-commit`).
- Added type hints (`Optional`, `Union`, `List`, `Tuple`, modern PEP 604 unions) across model, training, and prediction modules.

### Removed

- TensorFlow 1.x / Keras 2.x model definitions (`model_for_training`, `model_for_inference`) and the `compress_model` HDF5 utility.
- Legacy `csbdeep==0.6.0`, `bson`, `h5py==2.10.0`, and `tensorflow-gpu` pins.
- The `--max_patch` flag (renamed to `--max_examples_per_tile`).

### Known caveats / not yet updated

- `requirements.txt` and `Dockerfile` still reflect the original TensorFlow 1.14 stack and are **out of date** with the rest of the repository. Use `environment.yml` (conda) until they are refreshed; the Docker image will need to be rebuilt against a PyTorch base.
- The `tests/` directory and its `test_panels.py` suite from the upstream repository are not present in this fork.
- The optional `n_rays` polygon-distance head and the `ResImmuNet` backbone are scaffolded but commented out (`TODO: enable RAYS`).

---

## [1.0.0] — Original upstream release (reference)

The TensorFlow 1.14 / Keras 2.3 implementation accompanying:

> Sultan, Gorris, Martynova, _et al._ "ImmuNet: A Segmentation-Free Machine Learning Pipeline for Immune Landscape Phenotyping in Tumors by Multiplex Imaging." _Biology Methods and Protocols_, 2025. doi:[10.1093/biomethods/bpae094](https://doi.org/10.1093/biomethods/bpae094)

Source: <https://github.com/jtextor/immunet>. Included here only as the baseline this fork modernizes.
