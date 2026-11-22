from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = "cpu"
SEED = 42

# Pengaturan ramah CPU
IMG_SIZE = 160       # Bisa pakai 160 untuk eksperimen cepat atau 224 untuk EfficientNet-B0
BATCH_SIZE = 16      # Aman untuk RAM & CPU; bisa disesuaikan jika perlu
NUM_WORKERS = 0      # Stabil di Windows/CPU
VAL_SPLIT = 0.15     # 15% dari Training dijadikan validation set