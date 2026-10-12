"""Experiment constants from the project plan (fixed before test scores)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

DATA_ROOT = Path(
    __import__("os").environ.get(
        "TRANSISTOR_DATA_ROOT",
        PROJECT_ROOT.parent / "data" / "transistor",
    )
)

RANDOM_SEED = 42
TRAIN_NORMAL_COUNT = 170
VAL_NORMAL_COUNT = 43
TEST_GOOD_COUNT = 60
TEST_DEFECT_COUNT = 40
DEFECTS_PER_TYPE = 10

INPUT_SIZE = 256
PIXEL_SCALE = 255.0

DEFECT_TYPES = ("bent_lead", "cut_lead", "damaged_case", "misplaced")
DEFECT_LABELS = {
    "bent_lead": "Bent lead",
    "cut_lead": "Cut lead",
    "damaged_case": "Damaged case",
    "misplaced": "Misplaced",
}

METHOD_AUTOENCODER = "autoencoder"
METHOD_EMBEDDING = "embedding"

ENCODER_CHANNELS = (32, 64, 128, 128)
BOTTLENECK_CHANNELS = 64
BOTTLENECK_CHANNELS_NARROW = 16

CONV_KERNEL = 4
CONV_STRIDE = 2
CONV_PADDING = 1

AE_LEARNING_RATE = 0.001
AE_BATCH_SIZE = 16
AE_MAX_EPOCHS = 40
AE_EARLY_STOP_PATIENCE = 5
AE_WEIGHT_DECAY = 0.0
AE_COPY_MAE_THRESHOLD = 0.005

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH_BANK_SIZE = 4096
THRESHOLD_QUANTILE = 0.95

BRIGHTNESS_FACTORS = (0.7, 1.3)

TIMING_WARMUP_IMAGES = 10

SPLIT_JSON = OUTPUT_DIR / "split.json"
AUTOENCODER_WEIGHTS = OUTPUT_DIR / "autoencoder.pt"
AE_LOSS_CSV = OUTPUT_DIR / "ae_loss.csv"
PATCH_BANK_FILE = OUTPUT_DIR / "bank.pt"
SCORES_RAW_CSV = OUTPUT_DIR / "scores_raw.csv"
SCORES_BRIGHTNESS_RAW_CSV = OUTPUT_DIR / "scores_brightness_raw.csv"
SCORES_CSV = OUTPUT_DIR / "scores.csv"
TIMING_JSON = OUTPUT_DIR / "timing.json"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
RUN_META_JSON = OUTPUT_DIR / "run_meta.json"
