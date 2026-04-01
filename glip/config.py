"""Project-wide defaults."""

LR = 1e-4
WEIGHT_DECAY = 1e-3
TEMPERATURE = 1.0
IMAGE_SIZE = 224
CROP_SIZE = 224
PROJECTION_DIM = 256
DROPOUT = 0.1
IMAGE_EMBED_DIM = 2048
MODEL_NAME = "resnet50"
IMAGE_ENCODER_CHECKPOINT = "/data/yujk/UNI2-h/pytorch_model.bin"
GENE_ENCODER = "projection"
SCFOUNDATION_REPO_DIR = "/data/yujk/scFoundation"
SCFOUNDATION_CHECKPOINT = "/data/yujk/models.ckpt"
SCFOUNDATION_KEY = "cell"
SCFOUNDATION_POOL_TYPE = "all"
SCFOUNDATION_TGTHIGHRES = "t4"
USE_SCRNA_LOSS = False
SCRNA_DATA_PATH = "/data/zhaoyh/DeSCENT/data/BRCA/single_cell/BRCA_train.symbol_mapped.h5ad"
SCRNA_LOSS_WEIGHT = 0.1
SCRNA_KNN_PERCENT = 0.1

CONTROL_FEATURE_PREFIXES = (
    "NegControlProbe_",
    "NegControlCodeword_",
    "BLANK_",
    "Blank-",
    "antisense_",
    "Unassigned",
)
