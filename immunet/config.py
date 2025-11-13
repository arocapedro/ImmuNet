from pathlib import Path

# Default paths
DATA_FOLDER = Path("data")
PANEL_FILE = DATA_FOLDER / "panels.json"
ANNOTATIONS_FOLDER = DATA_FOLDER / "annotations"
PREDICTION_FOLDER = DATA_FOLDER / "prediction"
IMAGES_FOLDER = DATA_FOLDER / "tilecache"
PREDICTION_FILE = "prediction.tsv"
PREDICTION_FILE_PATH = PREDICTION_FOLDER / PREDICTION_FILE
VAL_ANNOTATONS_PATH = ANNOTATIONS_FOLDER / "annotations_val.json.gz"
TRAIN_ANNOTATONS_PATH = ANNOTATIONS_FOLDER / "annotations_train.json.gz"
TRAIN_OUTPUT_PATH = Path("train_output")
MODEL_PATH = TRAIN_OUTPUT_PATH / "immunet.pth"
MODEL_CP_NAME = "model_cp"
MODEL_FINAL_NAME = "immunet"
EVALUATION_PATH = Path("evaluation")
INPUT_FOLDER = Path("demo_input")
INPUT_IMAGE = "components.tiff"
INPUT_IMAGE_PATH = INPUT_FOLDER / INPUT_IMAGE
INFERENCE_PATH = Path("demo_inference")

# CONSTANTS USED IN DIFFERENT MODULES
# ANNOTATIONS JSON KEYS
DATASET_KEY = "ds"
SLIDE_KEY = "slide"
TILE_KEY = "tile"
ID_KEY = "id"
TYPE_KEY = "type"
X_KEY = "x"
Y_KEY = "y"
PHENO_KEY = "positivity"
BG_KEY = "background"
PANEL_KEY = "panel"
SLIDES_KEY = "slides"
TILES_KEY = "tiles"
ANNOTATIONS_KEY = "annotations"
