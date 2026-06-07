# ImmuNet — PyTorch port

A modernized **PyTorch 2.5 / Python 3.12** implementation of ImmuNet, a segmentation-free machine-learning pipeline for detecting and phenotyping immune cells in multiplex immunohistochemistry images.

> This repository is a fork of the original [`jtextor/immunet`](https://github.com/jtextor/immunet) codebase, which was written in TensorFlow 1.14 / Keras 2.3 on Python 3.6. The model architecture and the underlying science are unchanged — the framework, training pipeline, and tooling have been rebuilt. See [`WHATS_NEW.md`](WHATS_NEW.md) for a one-page summary or [`CHANGELOG.md`](CHANGELOG.md) for the detailed diff.

The model itself is described in:

> Shabaz Sultan, Mark A. J. Gorris, Evgenia Martynova, Lieke van der Woude, Franka Buytenhuijs, Sandra van Wilpe, Kiek Verrijp, Carl G. Figdor, I. Jolanda M. de Vries, Johannes Textor.
> **ImmuNet: A Segmentation-Free Machine Learning Pipeline for Immune Landscape Phenotyping in Tumors by Multiplex Imaging.**
> _Biology Methods and Protocols_, 2025. doi: [10.1093/biomethods/bpae094](https://doi.org/10.1093/biomethods/bpae094)

ImmuNet detects and phenotypes immune cells directly, without segmentation, which makes it particularly suitable for dense tissue contexts such as solid tumors where segmentation-based phenotyping is brittle. A subset of the immunohistochemistry images, annotations, and the published trained model are available on [Zenodo](https://zenodo.org/records/15046015). For convenience, a PyTorch-converted version of the published checkpoint, **`immunet_zenodo.pth`**, ships in the root of this repository.

---

## What's new in this fork

- **Framework:** TensorFlow 1.14 / Keras 2.3 → **PyTorch 2.5.1** + torchvision 0.20.1
- **Runtime:** Python 3.6.9 → **Python 3.12.8**, NumPy 2.x stack
- **Backbones:** the original architecture is reimplemented as `ImmuNet`, joined by two new backbones — `UImmNet` (U-Net style) and `DAPImmuNet` (DAPI-only). All share a common `BaseImmunet` API.
- **Training loop:** explicit PyTorch loop with **TensorBoard** logging, **mixed-precision (AMP)**, **early stopping**, **branch/backbone freezing**, configurable losses (`MSE` / `BCE`) and per-channel weighting.
- **Devices:** CUDA, CPU, and **Apple Silicon (MPS)** are all first-class.
- **Augmentations:** modern `torchvision.transforms.v2` stack (random rotations/flips, brightness, Gaussian noise, channel blackout, random crops); multi-process sample preparation.
- **New modules:** `models_parts.py`, `train_utils.py`, `prediction.py`, `visualization.py`.
- **Pretrained weights:** PyTorch checkpoint **`immunet_zenodo.pth`** is included in the repo. A `ImmuNet.from_keras_weights(...)` helper is also provided to convert the original Keras weights, should you have a custom-trained `.h5` file.
- **Model file format:** `.h5` → **`.pth`**.

The full diff is in [`CHANGELOG.md`](CHANGELOG.md).

---

## System requirements

The code has been tested on Python 3.12 with PyTorch 2.5.1 on Linux + CUDA, and on macOS with the **MPS** backend on Apple Silicon. CPU-only execution works but is slow for training. The architecture is unchanged from the paper, but PyTorch offered memory and runtime upgrades offering significant speed up (training the published model — 27,888 annotations, 100 epochs — took roughly 4 hours on a single RTX 2080 Ti).

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate immunet
```

The `environment.yml` pins PyTorch 2.5.1, torchvision 0.20.1, and a current numpy/scipy/pandas/scikit-* stack. Install the appropriate CUDA wheels for your system (`environment.yml` will pull the default CPU/CUDA build that pip resolves; users who need a specific CUDA toolkit should install PyTorch from the [official selector](https://pytorch.org/get-started/locally/) instead).

---

## Quick start

All commands below assume you are in the repository root.

### Inference on a single image

A pretrained PyTorch model (`immunet_zenodo.pth`) is shipped in the repo root. Place a `components.tiff` from a tile inside `demo_input/` and run:

```bash
python immunet/inference.py demo \
    --model_path immunet_zenodo.pth \
    --tile_path demo_input/components.tiff \
    --output_path demo_inference
```

This writes:

- `prediction.tsv` — predicted cell coordinates and per-marker phenotype scores
- `phenotype_prediction.png` — RGB visualization of the per-marker output, with detected cells overlaid
- `prediction_vis.jpg` — optional, when `--display_image_path` is provided: detected cells drawn on a custom display image (TIFF/PNG/JPEG)

If the input tile is too large for available GPU memory, the script automatically falls back to the central 512×512 sub-region. Reduce that further inside `crop_image_center(...)` if you still see OOM.

Useful flags:

| Flag | Purpose |
|---|---|
| `--log_th` | LoG blob-detection threshold |
| `--min_log_std` / `--max_log_std` | LoG sigma bounds (decrease/increase to detect smaller/larger cells) |
| `--vis_radius` | Radius of cell markers drawn on the visualization |

### Training

Download the data sample `tilecache.tar.gz` and the annotations `annotations_train.json.gz` from [Zenodo](https://zenodo.org/records/15046015). Place them so the layout is:

```
data/
  annotations/annotations_train.json.gz
  tilecache/...
```

Then run:

```bash
python immunet/train.py
```

By default this trains the original-architecture backbone with mixed precision off. Outputs (model checkpoint as `.pth`, TensorBoard event files, training history, run config) are written to `train_output/<YYYYMMDD-HHMMSS>_<experiment_keyword>/`.

Frequently used flags (run `python immunet/train.py --help` for the full list, organized into argument groups):

```bash
# Common knobs
--backbone {ORIGINAL,UNET,DAPI}     # architecture (default: ORIGINAL)
--epochs 500                        # default
--batch_size 64
--learning_rate 1e-3
--device_type {cpu,cuda,mps}        # CUDA / CPU / Apple-Silicon
--device_id 0
--autocast {on,off}                 # mixed-precision training
--loss_function {MSE,BCE}
--weighted_loss {on,off}            # Likert-scale-aware per-sample weighting
--early_stopper_patience 50
--early_stopper_delta 1e-3

# Data
--train_ann_path data/annotations/annotations_train.json.gz
--val_ann_path   data/annotations/annotations_val.json.gz
--images_path    data/tilecache
--cell_radius 5                     # label generation
--remove_channel <i>                # drop one or more input channels (repeatable)
--background_types "Other cell"     # override which annotation types are background

# Fine-tuning
--weights_path path/to/checkpoint.pth
--freeze_backbone | --freeze_distance | --freeze_phenotype

# UNet/DAPI knobs
--num_features 128                  # feature channels in the head
--base_channels 32                  # base width of the U-Net encoder

# Logging / experiment naming
--experiment_keyword my_run
--model_name current_best_model
--log_graph {on,off}                # log model graph to TensorBoard
```

Open TensorBoard with:

```bash
tensorboard --logdir train_output
```

### Evaluation

Place an annotation JSON in `data/annotations/`, the corresponding tiles in `data/tilecache/`, and a model in `train_output/`. Then run the full evaluation:

```bash
python immunet/evaluation.py run
```

This performs inference for every annotated tile, matches predictions against ground truth, and writes:

- `perf_maintypes.csv` and `perf_subtypes.csv` — error rates per cell type / subtype
- Confusion matrices for main types and subtypes
- An errors JSON listing each mismatched annotation, for case-by-case investigation

To only run the matching step (writing a `.tsv` of paired predictions and annotations), use:

```bash
python immunet/evaluation.py match --s <suffix>
```

Hyperparameters worth knowing:

- Matching step: same blob-detection flags as `inference.py` (`--log_th`, `--min_log_std`, `--max_log_std`)
- Performance step: `--marker_th` (per-marker activation threshold), `--radius` (matching radius in micrometers), `--pix_pmm` (pixels per micrometer)

---

## Loading models programmatically

```python
from immunet.models import load_model

model = load_model(
    device="cuda:0",            # or "cpu", "mps:0"
    which_model="immunet_zenodo.pth",
    backbone="ORIGINAL",        # or "UNET", "DAPI"
)
model.eval()
```

`BaseImmunet.from_path(...)` auto-detects the input-channel count and the number of phenotype markers from the saved state dict. To convert a Keras checkpoint trained with the original repository:

```python
from immunet.models import ImmuNet

# `keras_weights` is a dict[layer_name -> layer.get_weights()]
pytorch_model = ImmuNet.from_keras_weights(keras_weights)
```

---

## Using your own data

The dataset and annotation conventions are inherited from the upstream project and have **not** changed.

### Folder layout

```
- root folder
   - dataset1
      - slide1
         - tile1
            components.tiff
         - tile2
            ...
      - slide2
         ...
   - dataset2
      ...
```

### Annotation JSON

```json
[
  {
    "ds": "dataset1",
    "panel": "panel_name",
    "slides": [
      {
        "slide": "slide1",
        "tiles": [
          {
            "tile": "tile1",
            "annotations": [
              {
                "id": "5efdd2543de07424cab4fd3c",
                "type": "Other cell",
                "x": 1235,
                "y": 108,
                "positivity": [1, 1, 1, 1, 1],
                "background": true
              },
              {
                "id": "5efdd2543de07424cab4fd80",
                "type": "T cell",
                "x": 985,
                "y": 846,
                "positivity": [5, 5, 1, 1, 1]
              }
            ]
          }
        ]
      }
    ]
  }
]
```

`ds`, `slide`, and `tile` must match the folder names on disk. `panel` references a panel id defined in `data/panels.json`. `positivity` is a Likert-scale (1–5) vector of per-marker expression in the order the markers appear in the panel definition; for example with the lymphocyte panel below, `[5, 5, 1, 1, 1]` means CD3+, FOXP3+, CD20−, CD45RO−, CD8−. Background annotations should set `"background": true` and use `[1, 1, 1, 1, 1]`.

The image channels in `components.tiff` are typically `DAPI, CD3, FOXP3, CD20, CD45RO, CD8, Tumor marker, Autofluorescence`; only the marker channels enter the `positivity` vector.

### Defining panels (`data/panels.json`)

```json
{
  "panel": "lymphocyte",
  "markers": ["CD3", "FOXP3", "CD20", "CD45RO", "CD8"],
  "phenotypes": [
    {"type": "T cell", "subtype": "Thelp",   "phenotype": {"CD3": true,  "FOXP3": false, "CD20": false, "CD45RO": "*",  "CD8": false}},
    {"type": "T cell", "subtype": "Treg",    "phenotype": {"CD3": "*",   "FOXP3": true,  "CD20": false, "CD45RO": "*",  "CD8": false}},
    {"type": "T cell", "subtype": "Tcyt",    "phenotype": {"CD3": "*",   "FOXP3": false, "CD20": false, "CD45RO": "*",  "CD8": true }},
    {"type": "T cell", "subtype": "Tmemory", "phenotype": {"CD3": false, "FOXP3": false, "CD20": false, "CD45RO": true, "CD8": false}},
    {"type": "B cell",                       "phenotype": {"CD3": false, "FOXP3": false, "CD20": true,  "CD45RO": false,"CD8": false}},
    {"type": "Tumor cell", "background": true}
  ]
}
```

Each phenotype entry expects every panel marker as a key, with values `true`, `false`, or `"*"` (wildcard). Phenotypes with subtypes (e.g. T-cell variants) are repeated once per subtype. Phenotypes marked `"background": true` act as negative examples — `"No cell"` and `"Other cell"` are added automatically during JSON parsing.

### Inference over a whole dataset

There is no built-in script for whole-dataset inference. Iterate slide → tile, call `find_cells(...)` (or use the `ImmuNetPredictionHandler` class in `immunet/prediction.py`), and stitch the predictions together. Tile origins must come from the source TIFF metadata or the splitting tool you used to produce them.

---

## Repository layout

```
immunet/
  models.py            # BaseImmunet, ImmuNet, UImmNet, DAPImmuNet, load_model
  models_parts.py      # reusable nn.Module building blocks (NEW)
  train.py             # training entry point (PyTorch loop, TensorBoard, AMP)
  train_utils.py       # TrainConfig, EarlyStopper, AMP setup, factories (NEW)
  data_processing.py   # CustomImageDataset + torchvision.transforms.v2 augmentations
  prediction.py        # ImmuNetPredictionHandler, matching utilities (NEW)
  inference.py         # demo CLI for single-image inference
  evaluation.py        # `match` / `run` CLI for paired-prediction evaluation
  visualization.py     # confusion matrices, KDE/AUC plots, error bars (NEW)
  panels.py            # panel / phenotype definition parsing
  annotations.py       # Dataset / Slide / Tile loaders
  config.py            # default paths and JSON-key constants

data/
  panels.json          # panel definitions
  annotations/         # (you provide) annotation JSON files
  tilecache/           # (you provide) per-dataset/slide/tile imagery

paper/                 # reproduction scripts for the paper splits
scripts/docker/        # legacy docker helper scripts (TF era)
immunet_zenodo.pth     # converted PyTorch checkpoint of the published model
environment.yml        # conda environment for the PyTorch port
CHANGELOG.md           # full change log vs. upstream
WHATS_NEW.md           # one-page summary of the modernization
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{Sultan2025ImmuNet,
  author  = {Sultan, Shabaz and Gorris, Mark A. J. and Martynova, Evgenia and
             van der Woude, Lieke and Buytenhuijs, Franka and van Wilpe, Sandra and
             Verrijp, Kiek and Figdor, Carl G. and de Vries, I. Jolanda M. and
             Textor, Johannes},
  title   = {{ImmuNet}: A Segmentation-Free Machine Learning Pipeline for Immune
             Landscape Phenotyping in Tumors by Multiplex Imaging},
  journal = {Biology Methods and Protocols},
  year    = {2025},
  doi     = {10.1093/biomethods/bpae094}
}
```

## License

MIT, inherited from the upstream project (see [`LICENSE`](LICENSE)).

---

## Notice on AI-assisted documentation

The documentation files in this repository — this `README.md`, [`CHANGELOG.md`](CHANGELOG.md), and [`WHATS_NEW.md`](WHATS_NEW.md) — were drafted with the assistance of a large language model (Anthropic Claude), based on a diff between the upstream codebase and this fork. The **source code itself was written by humans**; no LLM was used to generate, modify, or refactor the model implementation, the training pipeline, or any other code in this repository. If you spot inaccuracies in the documentation, please open an issue — the code is authoritative.
