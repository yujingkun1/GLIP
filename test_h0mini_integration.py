#!/usr/bin/env python3
"""
Test script for H0-mini integration in GLIP.
Tests both pooled and patch_tokens modes.
"""

import torch
import sys
sys.path.insert(0, '/data/yujk/GLIP')

from glip.visium.modules import ImageEncoder_H0mini
from glip.visium.models import CLIPModel_H0mini
from glip.visium.patch_cell_matching import PatchCellMatcher
from shapely.geometry import Point

print("=" * 60)
print("Testing H0-mini Integration")
print("=" * 60)

# Test 1: ImageEncoder_H0mini - pooled mode
print("\n[Test 1] ImageEncoder_H0mini - pooled mode")
try:
    encoder_pooled = ImageEncoder_H0mini(
        output_mode="pooled",
        trainable=False,
        checkpoint_path="/data/yujk/H0-mini/pytorch_model.bin"
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    output_pooled = encoder_pooled(dummy_input)
    assert output_pooled.shape == (2, 768), f"Expected (2, 768), got {output_pooled.shape}"
    print(f"✓ Pooled mode output shape: {output_pooled.shape}")

    # Check parameters are frozen
    frozen_count = sum(1 for p in encoder_pooled.parameters() if not p.requires_grad)
    total_count = sum(1 for p in encoder_pooled.parameters())
    print(f"✓ Parameters frozen: {frozen_count}/{total_count}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: ImageEncoder_H0mini - patch_tokens mode
print("\n[Test 2] ImageEncoder_H0mini - patch_tokens mode")
try:
    encoder_patch = ImageEncoder_H0mini(
        output_mode="patch_tokens",
        trainable=False,
        checkpoint_path="/data/yujk/H0-mini/pytorch_model.bin"
    )
    output_patch = encoder_patch(dummy_input)
    assert output_patch.shape == (2, 256, 768), f"Expected (2, 256, 768), got {output_patch.shape}"
    print(f"✓ Patch tokens mode output shape: {output_patch.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: CLIPModel_H0mini
print("\n[Test 3] CLIPModel_H0mini")
try:
    model = CLIPModel_H0mini(
        spot_embedding=3467,
        pretrained=True,
        checkpoint_path="/data/yujk/H0-mini/pytorch_model.bin",
        output_mode="pooled",
        trainable=False,
    )
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "reduced_expression": torch.randn(4, 3467),
    }
    loss = model(batch)
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    print(f"✓ CLIPModel_H0mini forward pass successful")
    print(f"✓ Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: PatchCellMatcher
print("\n[Test 4] PatchCellMatcher")
try:
    matcher = PatchCellMatcher(image_size=224, patch_grid=16)
    patch_tokens = torch.randn(256, 768)

    # Create mock cell masks (circles)
    cell_masks = [
        Point(112, 112).buffer(20),  # Center cell
        Point(50, 50).buffer(15),    # Top-left cell
        Point(180, 180).buffer(18),  # Bottom-right cell
    ]

    # Test matching
    cell_features, indices, areas = matcher.match_patches_to_cells(
        patch_tokens,
        cell_masks,
        aggregation='weighted_mean'
    )

    assert cell_features.shape == (3, 768), f"Expected (3, 768), got {cell_features.shape}"
    assert len(indices) == 3, f"Expected 3 cells, got {len(indices)}"
    print(f"✓ Patch-cell matching successful")
    print(f"  Cell 0 overlaps with {len(indices[0])} patches")
    print(f"  Cell 1 overlaps with {len(indices[1])} patches")
    print(f"  Cell 2 overlaps with {len(indices[2])} patches")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Normalization parameters
print("\n[Test 5] Normalization parameters")
try:
    from glip.visium import config as visium_cfg
    from glip.xenium import config as xenium_cfg

    print(f"✓ Visium H0MINI_MEAN: {visium_cfg.H0MINI_MEAN}")
    print(f"✓ Visium H0MINI_STD: {visium_cfg.H0MINI_STD}")
    print(f"✓ Xenium H0MINI_MEAN: {xenium_cfg.H0MINI_MEAN}")
    print(f"✓ Xenium H0MINI_STD: {xenium_cfg.H0MINI_STD}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now train with H0-mini using:")
print("  python train_joint_brca_naive.py --model_name h0mini ...")
