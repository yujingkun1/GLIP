

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 128
# num_workers = 0
lr = 1e-4
weight_decay = 1e-3
patience = 5
factor = 0.5
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'uni'
image_embedding = 2048
spot_embedding = 3467 #number of shared hvgs (change for each dataset)

pretrained = True
trainable = True 
temperature = 1.0
image_encoder_checkpoint = "/data/yujk/UNI2-h/pytorch_model.bin"

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',

# H0-mini configuration
H0MINI_MODEL_NAME = "hf-hub:bioptimus/H0-mini"
H0MINI_CHECKPOINT = "/data/yujk/H0-mini/pytorch_model.bin"
H0MINI_OUTPUT_DIM = 768
H0MINI_OUTPUT_MODE = "pooled"  # 'pooled' for spot-level, 'patch_tokens' for cell-level

# H0-mini normalization parameters (different from ImageNet)
H0MINI_MEAN = [0.707223, 0.578729, 0.703617]
H0MINI_STD = [0.211883, 0.230117, 0.177517]

# ImageNet normalization (for other models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
