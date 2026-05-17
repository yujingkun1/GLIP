#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '/data/yujk/GLIP')

from glip.visium.modules import ImageEncoder_H0mini

encoder = ImageEncoder_H0mini(
    output_mode="pooled",
    trainable=False,
    checkpoint_path="/data/yujk/H0-mini/pytorch_model.bin"
)

dummy_input = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    output = encoder.model(dummy_input)
    print(f"Raw model output shape: {output.shape}")
    print(f"Raw model output type: {type(output)}")

    if hasattr(encoder.model, 'num_prefix_tokens'):
        print(f"num_prefix_tokens: {encoder.model.num_prefix_tokens}")

    # Try forward_features
    if hasattr(encoder.model, 'forward_features'):
        features = encoder.model.forward_features(dummy_input)
        print(f"forward_features output shape: {features.shape}")
