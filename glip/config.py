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

CONTROL_FEATURE_PREFIXES = (
    "NegControlProbe_",
    "NegControlCodeword_",
    "BLANK_",
    "Blank-",
    "antisense_",
    "Unassigned",
)
