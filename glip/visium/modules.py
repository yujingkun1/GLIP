import torch
from torch import nn
import timm

from . import config as CFG


UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"

MODEL_NAME_ALIASES = {
    "uni": UNI_MODEL_NAME,
    "uni2-h": UNI_MODEL_NAME,
}

UNI2_H_BACKBONE_NAME = "vit_giant_patch14_224"


def resolve_timm_model_name(model_name):
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError("Image encoder model name cannot be empty.")
    if normalized.startswith("hf_hub:"):
        normalized = "hf-hub:" + normalized[len("hf_hub:"):]
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def _infer_output_dim(model):
    output_dim = getattr(model, "num_features", None)
    if output_dim is not None:
        output_dim = int(output_dim)
        if output_dim > 0:
            return output_dim

    feature_info = getattr(model, "feature_info", None)
    if feature_info is not None and hasattr(feature_info, "channels"):
        channels = feature_info.channels()
        if channels:
            return int(channels[-1])

    raise AttributeError("Unable to infer the output dimension for the selected image encoder.")


def _load_local_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    model_state = model.state_dict()
    filtered_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            filtered_state_dict[clean_key] = value

    model.load_state_dict(filtered_state_dict, strict=False)


def _create_uni2_h_backbone(*, pretrained, checkpoint_path):
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    if pretrained and not checkpoint_path:
        return timm.create_model(UNI_MODEL_NAME, pretrained=True, **timm_kwargs)

    model = timm.create_model(
        UNI2_H_BACKBONE_NAME,
        pretrained=False,
        **timm_kwargs,
    )
    if checkpoint_path:
        _load_local_checkpoint(model, checkpoint_path)
    return model


def _create_timm_backbone(model_name, pretrained, checkpoint_path):
    resolved_model_name = resolve_timm_model_name(model_name)
    if resolved_model_name == UNI_MODEL_NAME:
        return _create_uni2_h_backbone(pretrained=pretrained, checkpoint_path=checkpoint_path)

    model = timm.create_model(
        resolved_model_name,
        pretrained=pretrained and not checkpoint_path,
        num_classes=0,
        global_pool="avg",
    )

    if checkpoint_path:
        _load_local_checkpoint(model, checkpoint_path)

    return model


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name=None,
        pretrained=None,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__()
        resolved_model_name = CFG.model_name if model_name is None else model_name
        resolved_pretrained = CFG.pretrained if pretrained is None else pretrained
        resolved_checkpoint = CFG.image_encoder_checkpoint if checkpoint_path is None else checkpoint_path

        self.model_name = resolve_timm_model_name(resolved_model_name)
        self.model = _create_timm_backbone(
            resolved_model_name,
            resolved_pretrained,
            resolved_checkpoint,
        )
        self.output_dim = _infer_output_dim(self.model)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet50(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name='resnet50',
        pretrained=None,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )
    
class ImageEncoder_resnet101(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name='resnet101',
        pretrained=True,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )
    
class ImageEncoder_resnet152(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name='resnet152',
        pretrained=True,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )
    
class ImageEncoder_ViT(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name="vit_base_patch32_224",
        pretrained=False,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )
    

class ImageEncoder_CLIP(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name="vit_base_patch32_224_clip_laion2b",
        pretrained=True,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )

class ImageEncoder_ViT_L(ImageEncoder):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name="vit_large_patch32_224_in21k",
        pretrained=False,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )


class ImageEncoder_UNI(ImageEncoder):
    """
    Encode images with MahmoodLab UNI2-h via timm + Hugging Face Hub.
    """

    def __init__(
        self,
        model_name=UNI_MODEL_NAME,
        pretrained=None,
        trainable=CFG.trainable,
        checkpoint_path=None,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            trainable=trainable,
            checkpoint_path=checkpoint_path,
        )


class ImageEncoder_H0mini(nn.Module):
    """
    Encode images with Bioptimus H0-mini (ViT-Base/14 pretrained on 43M histology tiles).
    Supports two output modes:
    - 'pooled': CLS token (batch_size, 768) for spot-level tasks
    - 'patch_tokens': 256 patch tokens (batch_size, 256, 768) for cell-level tasks
    """

    def __init__(
        self,
        pretrained=True,
        trainable=False,  # Default: freeze parameters
        checkpoint_path=CFG.H0MINI_CHECKPOINT,
        output_mode="pooled",  # 'pooled' or 'patch_tokens'
    ):
        super().__init__()
        self.output_mode = output_mode
        self.model_name = "h0mini"

        # Load H0-mini model from local directory
        # H0-mini uses SwiGLU activation with mlp_ratio=5.33334
        self.model = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=224,
            reg_tokens=4,
            dynamic_img_size=True,
            mlp_ratio=5.33334,  # H0-mini specific
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        # Load local checkpoint
        if pretrained and checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=True)

        # Set output dimension
        if output_mode == "pooled":
            self.output_dim = 768  # CLS token
        elif output_mode == "patch_tokens":
            self.output_dim = 768  # Each patch token dimension
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'pooled' or 'patch_tokens'")

        # Set trainability (default: frozen)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        # Use forward_features to get all tokens
        output = self.model.forward_features(x)  # (batch_size, 261, 768)
        # 261 tokens = 1 CLS + 4 reg + 256 patch tokens

        if self.output_mode == "pooled":
            return output[:, 0]  # CLS token: (batch_size, 768)
        elif self.output_mode == "patch_tokens":
            # Skip CLS and reg tokens, return only patch tokens
            num_prefix = self.model.num_prefix_tokens  # 5 (1 CLS + 4 reg)
            return output[:, num_prefix:]  # (batch_size, 256, 768)


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
    


# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())
            
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
