"""
Patch-Cell Matching Module for H0-mini

This module provides tools to match H0-mini patch tokens (16x16 grid) with cell masks,
enabling cell-level feature extraction from patch-level representations.
"""

import torch
import numpy as np
from shapely.geometry import box, Polygon
from shapely import wkb
import geopandas as gpd


class PatchCellMatcher:
    """
    Match patch tokens with cell masks for cell-level feature extraction.

    H0-mini outputs 256 patch tokens (16x16 grid) for a 224x224 image.
    Each patch corresponds to a 14x14 pixel region. This class computes
    which patches overlap with each cell mask and aggregates features.
    """

    def __init__(self, image_size=224, patch_grid=16):
        """
        Initialize the matcher.

        Args:
            image_size: Input image size (default: 224)
            patch_grid: Number of patches per dimension (default: 16)
        """
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.patch_size = image_size / patch_grid  # 14 pixels

        # Precompute all patch bounding boxes
        self.patch_bboxes = self._create_patch_bboxes()

    def _create_patch_bboxes(self):
        """Precompute bounding boxes for all patches."""
        bboxes = []
        for i in range(self.patch_grid):
            for j in range(self.patch_grid):
                bbox = box(
                    j * self.patch_size,
                    i * self.patch_size,
                    (j+1) * self.patch_size,
                    (i+1) * self.patch_size
                )
                bboxes.append(bbox)
        return bboxes

    def match_patches_to_cells(
        self,
        patch_tokens,
        cell_masks,
        aggregation='mean'
    ):
        """
        Match patch tokens to cell masks and aggregate features.

        Args:
            patch_tokens: Patch token features
                - Shape: (256, 768) or (batch_size, 256, 768)
            cell_masks: List of cell geometries
                - Can be Shapely Polygons, dicts with 'geometry', or WKB bytes
            aggregation: Aggregation method
                - 'mean': Simple average
                - 'max': Max pooling
                - 'weighted_mean': Area-weighted average

        Returns:
            cell_features: (num_cells, 768) - Aggregated features per cell
            cell_patch_indices: List[List[int]] - Patch indices for each cell
            overlap_areas: List[List[float]] - Overlap areas for each cell
        """
        # Handle batch dimension
        if patch_tokens.dim() == 3:
            batch_size = patch_tokens.shape[0]
            if batch_size != 1:
                raise ValueError("Batch matching not supported, process one image at a time")
            patch_tokens = patch_tokens[0]  # (256, 768)

        # Parse cell masks
        cell_polygons = self._parse_cell_masks(cell_masks)

        cell_features = []
        cell_patch_indices = []
        overlap_areas = []

        for cell_poly in cell_polygons:
            # Find overlapping patches
            overlapping_patches = []
            overlap_area_list = []

            for patch_idx, patch_bbox in enumerate(self.patch_bboxes):
                if cell_poly.intersects(patch_bbox):
                    overlapping_patches.append(patch_idx)
                    # Compute overlap area
                    intersection = cell_poly.intersection(patch_bbox)
                    overlap_area_list.append(intersection.area)

            # Aggregate patch tokens
            if overlapping_patches:
                selected_tokens = patch_tokens[overlapping_patches]  # (n_patches, 768)

                if aggregation == 'mean':
                    cell_feat = selected_tokens.mean(dim=0)
                elif aggregation == 'max':
                    cell_feat = selected_tokens.max(dim=0)[0]
                elif aggregation == 'weighted_mean':
                    # Weight by overlap area
                    weights = torch.tensor(overlap_area_list, dtype=torch.float32)
                    weights = weights / weights.sum()
                    weights = weights.to(selected_tokens.device)
                    cell_feat = (selected_tokens * weights.unsqueeze(1)).sum(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                # No overlap: return zero vector
                cell_feat = torch.zeros(768, device=patch_tokens.device)

            cell_features.append(cell_feat)
            cell_patch_indices.append(overlapping_patches)
            overlap_areas.append(overlap_area_list)

        return (
            torch.stack(cell_features),
            cell_patch_indices,
            overlap_areas
        )

    def _parse_cell_masks(self, cell_masks):
        """Parse cell masks from various formats."""
        polygons = []

        for mask in cell_masks:
            if isinstance(mask, Polygon):
                polygons.append(mask)
            elif isinstance(mask, dict) and 'geometry' in mask:
                # GeoJSON format
                geom = mask['geometry']
                if isinstance(geom, bytes):
                    # WKB format
                    polygons.append(wkb.loads(geom))
                else:
                    polygons.append(Polygon(geom['coordinates'][0]))
            elif isinstance(mask, bytes):
                # WKB format
                polygons.append(wkb.loads(mask))
            else:
                raise ValueError(f"Unsupported mask format: {type(mask)}")

        return polygons

    def load_cell_masks_from_parquet(self, parquet_path):
        """Load cell masks from a parquet file."""
        gdf = gpd.read_parquet(parquet_path)
        return gdf['geometry'].tolist()

    def visualize_matching(self, patch_tokens, cell_masks, save_path=None):
        """
        Visualize patch-cell matching.

        Args:
            patch_tokens: Patch tokens (not used for visualization)
            cell_masks: Cell geometries
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Polygon as MPLPolygon

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw patch grid
        for i in range(self.patch_grid):
            for j in range(self.patch_grid):
                rect = Rectangle(
                    (j * self.patch_size, i * self.patch_size),
                    self.patch_size, self.patch_size,
                    fill=False, edgecolor='gray', linewidth=0.5
                )
                ax.add_patch(rect)

        # Draw cell masks
        cell_polygons = self._parse_cell_masks(cell_masks)
        for poly in cell_polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, 'b-', linewidth=1)

        ax.set_xlim(0, self.image_size)
        ax.set_ylim(0, self.image_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Patch-Cell Matching Visualization')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def match_patches_to_cells(
    patch_tokens,
    cell_masks,
    image_size=224,
    patch_grid=16,
    aggregation='mean'
):
    """
    Convenience function to match patch tokens with cell masks.

    Args:
        patch_tokens: (256, 768) or (1, 256, 768)
        cell_masks: List of cell geometries
        image_size: Image size (default: 224)
        patch_grid: Patch grid size (default: 16)
        aggregation: Aggregation method (default: 'mean')

    Returns:
        cell_features: (num_cells, 768)
        cell_patch_indices: List[List[int]]
        overlap_areas: List[List[float]]
    """
    matcher = PatchCellMatcher(image_size, patch_grid)
    return matcher.match_patches_to_cells(patch_tokens, cell_masks, aggregation)
