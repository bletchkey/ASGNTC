from pathlib import Path

PROJECT_NAME = "ASGNTC"

ROOT_DIR = Path(__file__).resolve().parents[1]

MEDIA_DIR = Path("/media/yui/bletchkey")

CONFIG_DIR         = ROOT_DIR / "configs"
SRC_DIR            = ROOT_DIR / "src"

DATASET_DIR        = MEDIA_DIR / PROJECT_NAME / "data"
TRAININGS_DIR      = MEDIA_DIR / PROJECT_NAME / "trainings"
TRAINED_MODELS_DIR = MEDIA_DIR / PROJECT_NAME / "trained_models"

