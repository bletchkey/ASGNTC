from pathlib import Path

PROJECT_NAME = "ASGNTC"

ROOT_DIR = Path(__file__).resolve().parents[1]

MEDIA_DIR = Path("/media/yui/bletchkey")

CONFIG_DIR         = ROOT_DIR / "configs"
SRC_DIR            = ROOT_DIR / "src"
OUTPUTS_DIR        = ROOT_DIR / "outputs"

DATASET_DIR        = MEDIA_DIR / PROJECT_NAME / "data"
TRAININGS_DIR      = MEDIA_DIR / PROJECT_NAME / "trainings"

TRAININGS_PREDICTOR_DIR   = TRAININGS_DIR / "predictor"
TRAININGS_ADVERSARIAL_DIR = TRAININGS_DIR / "adversarial"

TRAINED_MODELS_DIR = MEDIA_DIR / PROJECT_NAME / "trained_models"

APP_LOGS_DIR        = MEDIA_DIR / PROJECT_NAME / "app_logs"
