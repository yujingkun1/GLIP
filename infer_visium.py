#!/usr/bin/env python3
"""Run Xenium-trained GLIP checkpoints on HEST Visium/ST samples via cell-to-spot aggregation."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import geopandas as gpd
import numpy as np
import openslide
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from glip.xenium import config as CFG
from glip.xenium.data import XeniumSingleCellDataset, _SimpleWSI, load_gene_names_from_tsv, prepare_processed_dataset
from glip.xenium.model import ContrastiveImageGeneModel, resolve_image_model_name
from glip.utils import compute_pearson_metrics, parse_bool, safe_pearson, sample_indices, save_json


def create_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer Visium/ST spot expression from a Xenium-trained GLIP checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        default="/data/yujk/GLIP/runs_xenium/ncbi784_uni/best_train_loss.pt",
        help="Path to a trained GLIP checkpoint",
    )
    parser.add_argument(
        "--hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data",
        help="HEST Visium/ST root directory",
    )
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("--sample-id", help="HEST sample id to evaluate, e.g. MEND141 or SPA124")
    sample_group.add_argument(
        "--sample-ids",
        help="Comma-separated HEST sample ids to evaluate in one run",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/runs_xenium/visium_inference",
        help="Directory for inference outputs",
    )
    parser.add_argument("--device", default="", help="Torch device, empty means auto")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for Visium cell image inference")
    parser.add_argument("--bank-batch-size", type=int, default=256, help="Batch size for Xenium retrieval bank embedding")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k Xenium neighbors used for cell-level expression retrieval")
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024, help="Chunk size for retrieval similarity")
    parser.add_argument("--max-cells", type=int, default=0, help="Optional cap on assigned Visium cells, 0 means all")
    parser.add_argument("--max-spots", type=int, default=0, help="Optional cap on evaluated Visium spots, 0 means all")
    parser.add_argument("--bank-max-cells", type=int, default=0, help="Optional cap on Xenium bank cells, 0 means all")
    parser.add_argument(
        "--in-tissue-only",
        default="true",
        help="Whether to evaluate only in-tissue spots when the HEST AnnData provides that flag",
    )
    parser.add_argument(
        "--assignment-radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the metadata-derived spot radius during cell-to-spot assignment",
    )
    parser.add_argument(
        "--aggregation",
        choices=["sum"],
        default="sum",
        help="Bag aggregation from predicted cells to spots; sum means sum predicted counts then log1p",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when sampling max cells/spots")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_sample_id_list(sample_id: Optional[str], sample_ids: Optional[str]) -> List[str]:
    if sample_ids:
        parsed = [item.strip() for item in str(sample_ids).split(",") if item.strip()]
        if parsed:
            return parsed
    if sample_id:
        return [str(sample_id).strip()]
    raise ValueError("Either --sample-id or --sample-ids must be provided")


def parse_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def resolve_cellvit_seg_path(hest_data_dir: str, sample_id: str) -> str:
    cellvit_dir = os.path.join(hest_data_dir, "cellvit_seg")
    candidates = [
        os.path.join(cellvit_dir, f"{sample_id}_cellvit_seg.parquet"),
        os.path.join(cellvit_dir, f"{sample_id}_cellvit_seg.geojson"),
        os.path.join(cellvit_dir, f"{sample_id}_cellvit_seg.geojson.zip"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to find CellViT segmentation for {sample_id} under {cellvit_dir}")


def load_cellvit_segmentation(seg_path: str) -> gpd.GeoDataFrame:
    if seg_path.endswith(".parquet"):
        return gpd.read_parquet(seg_path)
    if seg_path.endswith(".geojson"):
        return gpd.read_file(seg_path)
    if seg_path.endswith(".zip"):
        with zipfile.ZipFile(seg_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".geojson")]
            if not members:
                raise FileNotFoundError(f"No .geojson file found inside {seg_path}")
            member = members[0]
            with tempfile.TemporaryDirectory(prefix="cellvit_geojson_") as tmpdir:
                extracted_path = archive.extract(member, path=tmpdir)
                return gpd.read_file(extracted_path)
    raise ValueError(f"Unsupported CellViT segmentation format: {seg_path}")


@dataclass
class SpotData:
    sample_id: str
    technology: str
    spot_barcodes: np.ndarray
    spot_centers: np.ndarray
    spot_expression_log1p: np.ndarray
    gene_names: List[str]
    spot_radius_px: float
    pixel_size_um: float
    target_space: str
    count_semantics_verified: bool


def verify_counts_like_expression(adata: ad.AnnData, sample_id: str) -> bool:
    row_count = min(int(adata.n_obs), 256)
    col_count = min(int(adata.n_vars), 256)
    block = adata.X[:row_count, :col_count]
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float32)
    nonzero = block[block > 0]
    if nonzero.size == 0:
        return True

    non_integer_frac = float(np.mean(np.abs(nonzero - np.round(nonzero)) > 1e-6))
    if non_integer_frac > 0.001:
        raise ValueError(
            f"{sample_id} spot expression does not look like raw counts "
            f"(sampled non-integer fraction={non_integer_frac:.4f}). "
            "Current evaluator expects count-like spot expression so it can compare in log1p(count) space."
        )
    return True


def load_visium_spot_data(
    *,
    hest_data_dir: str,
    sample_id: str,
    target_gene_names: Sequence[str],
    in_tissue_only: bool,
    max_spots: int,
    seed: int,
) -> SpotData:
    st_path = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
    metadata_path = os.path.join(hest_data_dir, "metadata", f"{sample_id}.json")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"HEST spot AnnData not found: {st_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"HEST metadata JSON not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    pixel_size_um = parse_optional_float(metadata.get("pixel_size_um_embedded"))
    if pixel_size_um is None:
        pixel_size_um = parse_optional_float(metadata.get("pixel_size_um_estimated"))
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(f"Unable to resolve pixel size for {sample_id} from {metadata_path}")

    spot_diameter_um = parse_optional_float(metadata.get("spot_diameter"))
    if spot_diameter_um is None or spot_diameter_um <= 0:
        raise ValueError(f"Unable to resolve spot diameter for {sample_id} from {metadata_path}")

    adata = ad.read_h5ad(st_path, backed="r")
    try:
        count_semantics_verified = verify_counts_like_expression(adata, sample_id)
        spot_mask = np.ones(adata.n_obs, dtype=bool)
        if in_tissue_only and "in_tissue" in adata.obs.columns:
            spot_mask &= adata.obs["in_tissue"].to_numpy().astype(np.int64, copy=False) > 0

        valid_spot_indices = np.flatnonzero(spot_mask)
        if max_spots and valid_spot_indices.size > max_spots:
            sampled = sample_indices(total_size=valid_spot_indices.size, max_items=max_spots, seed=seed)
            valid_spot_indices = valid_spot_indices[sampled]

        overlap_gene_names = [gene_name for gene_name in target_gene_names if gene_name in adata.var_names]
        if not overlap_gene_names:
            raise RuntimeError(f"No overlapping genes between Xenium panel and {sample_id}")

        gene_indexer = adata.var_names.get_indexer(overlap_gene_names)
        expression = adata.X[valid_spot_indices][:, gene_indexer]
        if hasattr(expression, "toarray"):
            expression = expression.toarray()
        expression = np.asarray(expression, dtype=np.float32)
        expression = np.log1p(np.clip(expression, a_min=0.0, a_max=None))

        spot_centers = np.asarray(adata.obsm["spatial"][valid_spot_indices], dtype=np.float32)
        spot_barcodes = np.asarray([str(adata.obs_names[idx]) for idx in valid_spot_indices], dtype=object)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

        technology = str(metadata.get("st_technology") or "UNKNOWN")
    return SpotData(
        sample_id=sample_id,
        technology=technology,
        spot_barcodes=spot_barcodes,
        spot_centers=spot_centers,
        spot_expression_log1p=expression,
        gene_names=overlap_gene_names,
        spot_radius_px=float(spot_diameter_um / pixel_size_um / 2.0),
        pixel_size_um=float(pixel_size_um),
        target_space="log1p_counts",
        count_semantics_verified=bool(count_semantics_verified),
    )


@dataclass
class CellAssignment:
    cell_ids: np.ndarray
    centroids: np.ndarray
    spot_indices: np.ndarray


def assign_cells_to_spots(
    *,
    hest_data_dir: str,
    sample_id: str,
    spot_centers: np.ndarray,
    spot_radius_px: float,
    radius_scale: float,
    max_cells: int,
    seed: int,
) -> CellAssignment:
    seg_path = resolve_cellvit_seg_path(hest_data_dir, sample_id)
    seg_gdf = load_cellvit_segmentation(seg_path)
    if "geometry" not in seg_gdf.columns:
        raise KeyError(f"CellViT segmentation missing geometry column: {seg_path}")

    centroids = seg_gdf.geometry.centroid
    centroid_xy = np.column_stack([centroids.x.to_numpy(dtype=np.float32), centroids.y.to_numpy(dtype=np.float32)])
    if "cell_id" in seg_gdf.columns:
        cell_ids = seg_gdf["cell_id"].to_numpy(dtype=np.int64, copy=False)
    else:
        cell_ids = np.arange(len(seg_gdf), dtype=np.int64)

    tree = cKDTree(np.asarray(spot_centers, dtype=np.float32))
    max_distance = float(spot_radius_px * max(float(radius_scale), 1e-6))
    distances, nearest = tree.query(centroid_xy, distance_upper_bound=max_distance)
    valid_mask = np.isfinite(distances) & (nearest < spot_centers.shape[0])
    valid_indices = np.flatnonzero(valid_mask)
    if max_cells and valid_indices.size > max_cells:
        sampled = sample_indices(total_size=valid_indices.size, max_items=max_cells, seed=seed)
        valid_indices = valid_indices[sampled]

    return CellAssignment(
        cell_ids=cell_ids[valid_indices],
        centroids=centroid_xy[valid_indices],
        spot_indices=nearest[valid_indices].astype(np.int64, copy=False),
    )


class VisiumCellInferenceDataset(Dataset):
    def __init__(
        self,
        *,
        hest_data_dir: str,
        sample_id: str,
        cell_ids: np.ndarray,
        centroids: np.ndarray,
        spot_indices: np.ndarray,
        crop_size: int,
        image_size: int,
        wsi_level: int,
    ) -> None:
        super().__init__()
        self.hest_data_dir = os.path.abspath(hest_data_dir)
        self.sample_id = sample_id
        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.spot_indices = np.asarray(spot_indices, dtype=np.int64)
        self.crop_size = int(crop_size)
        self.image_size = int(image_size)
        self.wsi_level = int(wsi_level)
        self.wsi_path = os.path.join(self.hest_data_dir, "wsis", f"{sample_id}.tif")
        if not os.path.exists(self.wsi_path):
            raise FileNotFoundError(f"HEST WSI not found: {self.wsi_path}")
        self._slide = None

    def __len__(self) -> int:
        return int(self.cell_ids.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def _get_slide(self):
        if self._slide is None:
            try:
                self._slide = openslide.OpenSlide(self.wsi_path)
            except Exception:
                self._slide = _SimpleWSI(self.wsi_path)
        return self._slide

    def _read_crop(self, center_x: float, center_y: float) -> Image.Image:
        slide = self._get_slide()
        downsample = float(slide.level_downsamples[self.wsi_level])
        read_size = max(1, int(round(self.crop_size / downsample)))
        left = int(round(float(center_x) - self.crop_size / 2.0))
        top = int(round(float(center_y) - self.crop_size / 2.0))
        region = slide.read_region((left, top), self.wsi_level, (read_size, read_size)).convert("RGB")
        if region.size != (self.image_size, self.image_size):
            region = region.resize((self.image_size, self.image_size), Image.BILINEAR)
        return region

    @staticmethod
    def _transform(image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        center_x = float(self.centroids[index, 0])
        center_y = float(self.centroids[index, 1])
        image = self._transform(self._read_crop(center_x, center_y))
        return {
            "image": image.float(),
            "cell_id": int(self.cell_ids[index]),
            "spot_index": int(self.spot_indices[index]),
        }

    def close(self) -> None:
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass
            self._slide = None

    def __del__(self) -> None:
        self.close()


def collect_bank_embeddings(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    embeddings: List[torch.Tensor] = []
    expressions: List[torch.Tensor] = []
    cell_ids: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="xenium_bank", leave=False):
            expressions_batch = batch.get("encoder_expression", batch["expression"]).to(device, non_blocking=True)
            batch_embeddings = model.encode_genes(expressions_batch)
            embeddings.append(batch_embeddings.detach().cpu())
            expressions.append(batch["expression"].detach().cpu())
            cell_ids.append(torch.as_tensor(batch["cell_id"], dtype=torch.int64))

    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "expressions": torch.cat(expressions, dim=0),
        "cell_ids": torch.cat(cell_ids, dim=0),
    }


def collect_query_embeddings(
    model: ContrastiveImageGeneModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    embeddings: List[torch.Tensor] = []
    cell_ids: List[torch.Tensor] = []
    spot_indices: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="visium_cells", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            batch_embeddings = model.encode_images(images)
            embeddings.append(batch_embeddings.detach().cpu())
            cell_ids.append(torch.as_tensor(batch["cell_id"], dtype=torch.int64))
            spot_indices.append(torch.as_tensor(batch["spot_index"], dtype=torch.int64))

    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "cell_ids": torch.cat(cell_ids, dim=0),
        "spot_indices": torch.cat(spot_indices, dim=0),
    }


def predict_expression_from_retrieval(
    *,
    bank: Dict[str, torch.Tensor],
    queries: Dict[str, torch.Tensor],
    top_k: int,
    chunk_size: int,
) -> np.ndarray:
    bank_embeddings = F.normalize(bank["embeddings"].float(), dim=1)
    bank_expressions = bank["expressions"].float()
    query_embeddings = F.normalize(queries["embeddings"].float(), dim=1)

    if bank_embeddings.size(0) == 0:
        raise RuntimeError("Retrieval bank is empty.")

    effective_top_k = max(1, min(int(top_k), bank_embeddings.size(0)))
    predictions: List[torch.Tensor] = []

    for start in range(0, query_embeddings.size(0), max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), query_embeddings.size(0))
        query_chunk = query_embeddings[start:end]
        similarity = query_chunk @ bank_embeddings.T

        top_values, top_indices = similarity.topk(effective_top_k, dim=1)
        matched = bank_expressions.index_select(0, top_indices.reshape(-1)).view(query_chunk.size(0), effective_top_k, -1)

        if effective_top_k == 1:
            chunk_prediction = matched[:, 0, :]
        else:
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            chunk_prediction = (matched * weights).sum(dim=1)
        predictions.append(chunk_prediction.cpu())

    return torch.cat(predictions, dim=0).numpy()


def aggregate_cells_to_spots(
    *,
    cell_predictions_log1p: np.ndarray,
    cell_spot_indices: np.ndarray,
    num_spots: int,
) -> Tuple[np.ndarray, np.ndarray]:
    predicted_counts = np.expm1(np.asarray(cell_predictions_log1p, dtype=np.float32))
    spot_counts = np.zeros((num_spots, predicted_counts.shape[1]), dtype=np.float32)
    cells_per_spot = np.zeros(num_spots, dtype=np.int64)
    np.add.at(spot_counts, cell_spot_indices, predicted_counts)
    np.add.at(cells_per_spot, cell_spot_indices, 1)
    return np.log1p(spot_counts), cells_per_spot


def compute_gene_pearsons(predictions: np.ndarray, targets: np.ndarray, gene_names: Sequence[str]) -> Dict[str, float]:
    return {
        str(gene_name): float(safe_pearson(predictions[:, gene_idx], targets[:, gene_idx]))
        for gene_idx, gene_name in enumerate(gene_names)
    }


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ContrastiveImageGeneModel, Dict, List[str], bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = dict(checkpoint.get("args", {}))

    sample_id = checkpoint_args.get("sample_id", "NCBI784")
    processed_dir = checkpoint_args.get("processed_dir", "/data/yujk/GLIP/processed")
    hest_data_dir = checkpoint_args.get("hest_data_dir", "/data/yujk/hovernet2feature/HEST/hest_data_Xenium")

    processed_paths = prepare_processed_dataset(
        hest_data_dir=hest_data_dir,
        output_dir=processed_dir,
        sample_id=sample_id,
        remove_control_features=parse_bool(checkpoint_args.get("remove_control_features", True)),
        nucleus_only=parse_bool(checkpoint_args.get("nucleus_only", False)),
        drop_zero_expression=parse_bool(checkpoint_args.get("drop_zero_expression", True)),
        force_rebuild=False,
    )

    gene_encoder = str(checkpoint_args.get("gene_encoder") or CFG.GENE_ENCODER).strip().lower()
    if gene_encoder == "scfoundation":
        scfoundation_repo_dir = checkpoint_args.get("scfoundation_repo_dir") or CFG.SCFOUNDATION_REPO_DIR
        encoder_target_gene_names = load_gene_names_from_tsv(
            os.path.join(scfoundation_repo_dir, "OS_scRNA_gene_index.19264.tsv")
            if os.path.exists(os.path.join(scfoundation_repo_dir, "OS_scRNA_gene_index.19264.tsv"))
            else os.path.join(scfoundation_repo_dir, "model", "OS_scRNA_gene_index.19264.tsv")
        )
        encoder_use_raw_counts = True
    else:
        with open(processed_paths.genes_path, "r", encoding="utf-8") as handle:
            encoder_target_gene_names = list(json.load(handle)["genes"])
        encoder_use_raw_counts = False

    resolved_model_name = checkpoint_args.get("resolved_model_name") or resolve_image_model_name(
        checkpoint_args.get("model", CFG.MODEL_NAME)
    )
    model = ContrastiveImageGeneModel(
        gene_dim=len(encoder_target_gene_names),
        model_name=resolved_model_name,
        pretrained=False,
        image_encoder_checkpoint="",
        temperature=float(checkpoint_args.get("temperature", CFG.TEMPERATURE)),
        gene_encoder=gene_encoder,
        scfoundation_repo_dir=checkpoint_args.get("scfoundation_repo_dir") or CFG.SCFOUNDATION_REPO_DIR,
        scfoundation_checkpoint=checkpoint_args.get("scfoundation_checkpoint") or CFG.SCFOUNDATION_CHECKPOINT,
        scfoundation_key=checkpoint_args.get("scfoundation_key") or CFG.SCFOUNDATION_KEY,
        scfoundation_pool_type=checkpoint_args.get("scfoundation_pool_type") or CFG.SCFOUNDATION_POOL_TYPE,
        scfoundation_tgthighres=checkpoint_args.get("scfoundation_tgthighres") or CFG.SCFOUNDATION_TGTHIGHRES,
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch. Missing keys: {missing_keys[:10]}, unexpected keys: {unexpected_keys[:10]}"
        )
    model.eval()

    checkpoint_context = {
        "sample_id": sample_id,
        "processed_dir": processed_dir,
        "hest_data_dir": hest_data_dir,
        "test_fold": int(checkpoint_args.get("test_fold", 4)),
        "num_position_folds": int(checkpoint_args.get("num_position_folds", 5)),
        "crop_size": int(checkpoint_args.get("crop_size", CFG.CROP_SIZE)),
        "image_size": int(checkpoint_args.get("image_size", CFG.IMAGE_SIZE)),
        "wsi_level": int(checkpoint_args.get("wsi_level", 0)),
        "resolved_model_name": resolved_model_name,
        "gene_encoder": gene_encoder,
    }
    return model, checkpoint_context, encoder_target_gene_names, encoder_use_raw_counts


def run_sample_inference(
    *,
    sample_id: str,
    args: argparse.Namespace,
    model: ContrastiveImageGeneModel,
    checkpoint_context: Dict,
    xenium_gene_names: List[str],
    encoder_use_raw_counts: bool,
    bank: Dict[str, torch.Tensor],
) -> Dict:
    spot_data = load_visium_spot_data(
        hest_data_dir=args.hest_data_dir,
        sample_id=sample_id,
        target_gene_names=xenium_gene_names,
        in_tissue_only=parse_bool(args.in_tissue_only),
        max_spots=args.max_spots,
        seed=args.seed,
    )
    cell_assignment = assign_cells_to_spots(
        hest_data_dir=args.hest_data_dir,
        sample_id=sample_id,
        spot_centers=spot_data.spot_centers,
        spot_radius_px=spot_data.spot_radius_px,
        radius_scale=args.assignment_radius_scale,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    if cell_assignment.cell_ids.size == 0:
        raise RuntimeError(f"No CellViT cells were assigned to any selected spot for sample {sample_id}")

    visium_cell_dataset = VisiumCellInferenceDataset(
        hest_data_dir=args.hest_data_dir,
        sample_id=sample_id,
        cell_ids=cell_assignment.cell_ids,
        centroids=cell_assignment.centroids,
        spot_indices=cell_assignment.spot_indices,
        crop_size=checkpoint_context["crop_size"],
        image_size=checkpoint_context["image_size"],
        wsi_level=checkpoint_context["wsi_level"],
    )
    query_loader = create_loader(visium_cell_dataset, args.batch_size, args.num_workers, shuffle=False)
    queries = collect_query_embeddings(model, query_loader, resolve_device(args.device))

    cell_predictions = predict_expression_from_retrieval(
        bank=bank,
        queries=queries,
        top_k=args.top_k,
        chunk_size=args.retrieval_chunk_size,
    )

    target_gene_set = set(spot_data.gene_names)
    overlap_gene_positions = np.asarray(
        [idx for idx, gene_name in enumerate(xenium_gene_names) if gene_name in target_gene_set],
        dtype=np.int64,
    )
    overlap_gene_names = [xenium_gene_names[idx] for idx in overlap_gene_positions.tolist()]
    target_gene_index = {gene_name: idx for idx, gene_name in enumerate(spot_data.gene_names)}
    target_gene_positions = np.asarray([target_gene_index[gene_name] for gene_name in overlap_gene_names], dtype=np.int64)

    spot_predictions_log1p, cells_per_spot = aggregate_cells_to_spots(
        cell_predictions_log1p=cell_predictions[:, overlap_gene_positions],
        cell_spot_indices=queries["spot_indices"].cpu().numpy().astype(np.int64, copy=False),
        num_spots=spot_data.spot_centers.shape[0],
    )
    target_spot_expression = spot_data.spot_expression_log1p[:, target_gene_positions]

    covered_mask = cells_per_spot > 0
    if not covered_mask.any():
        raise RuntimeError(f"No spots received any assigned cells for sample {sample_id}")

    covered_predictions = spot_predictions_log1p[covered_mask]
    covered_targets = target_spot_expression[covered_mask]
    metrics = compute_pearson_metrics(covered_predictions, covered_targets, entity_label="spot")
    gene_pearsons = compute_gene_pearsons(covered_predictions, covered_targets, overlap_gene_names)

    return {
        "sample_id": sample_id,
        "technology": spot_data.technology,
        "device": str(resolve_device(args.device)),
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "checkpoint_sample_id": checkpoint_context["sample_id"],
        "xenium_bank_cells": int(bank["embeddings"].shape[0]),
        "visium_cells_assigned": int(cell_assignment.cell_ids.shape[0]),
        "evaluated_spots": int(int(covered_mask.sum())),
        "candidate_spots": int(spot_data.spot_centers.shape[0]),
        "spot_coverage": float(float(covered_mask.mean())),
        "spot_radius_px": float(spot_data.spot_radius_px),
        "pixel_size_um": float(spot_data.pixel_size_um),
        "assignment_radius_scale": float(args.assignment_radius_scale),
        "top_k": int(args.top_k),
        "retrieval_chunk_size": int(args.retrieval_chunk_size),
        "aggregation": args.aggregation,
        "target_space": spot_data.target_space,
        "count_semantics_verified": bool(spot_data.count_semantics_verified),
        "num_overlap_genes": int(len(overlap_gene_names)),
        "overlap_genes": overlap_gene_names,
        "cells_per_spot_mean": float(cells_per_spot[covered_mask].mean()),
        "cells_per_spot_std": float(cells_per_spot[covered_mask].std()),
        "metrics": metrics,
        "gene_pearsons": gene_pearsons,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    sample_ids = parse_sample_id_list(args.sample_id, args.sample_ids)

    model, checkpoint_context, xenium_gene_names, encoder_use_raw_counts = load_checkpoint_model(
        args.checkpoint_path,
        device,
    )

    train_bank_dataset = XeniumSingleCellDataset(
        processed_dir=checkpoint_context["processed_dir"],
        hest_data_dir=checkpoint_context["hest_data_dir"],
        sample_id=checkpoint_context["sample_id"],
        split="train",
        test_fold=checkpoint_context["test_fold"],
        num_position_folds=checkpoint_context["num_position_folds"],
        crop_size=checkpoint_context["crop_size"],
        image_size=checkpoint_context["image_size"],
        wsi_level=checkpoint_context["wsi_level"],
        augment=False,
        include_image=False,
        max_cells=args.bank_max_cells,
        encoder_target_gene_names=xenium_gene_names,
        encoder_use_raw_counts=encoder_use_raw_counts,
    )
    bank_loader = create_loader(train_bank_dataset, args.bank_batch_size, args.num_workers, shuffle=False)
    bank = collect_bank_embeddings(model, bank_loader, device)

    sample_payloads: List[Dict] = []
    for sample_id in sample_ids:
        output_payload = run_sample_inference(
            sample_id=sample_id,
            args=args,
            model=model,
            checkpoint_context=checkpoint_context,
            xenium_gene_names=xenium_gene_names,
            encoder_use_raw_counts=encoder_use_raw_counts,
            bank=bank,
        )
        sample_output_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        save_json(output_payload, os.path.join(sample_output_dir, "metrics.json"))
        sample_payloads.append(output_payload)

        metrics = output_payload["metrics"]
        print(f"Sample: {sample_id} ({output_payload['technology']})")
        print(f"Assigned cells: {output_payload['visium_cells_assigned']}")
        print(f"Covered spots: {output_payload['evaluated_spots']}/{output_payload['candidate_spots']}")
        print(f"Overlap genes: {output_payload['num_overlap_genes']}")
        print(
            "Pearson: overall={overall:.4f}, mean_gene={gene:.4f}, mean_spot={spot:.4f}".format(
                overall=metrics["overall_pearson"],
                gene=metrics["mean_gene_pearson"],
                spot=metrics["mean_spot_pearson"],
            )
        )
        print(f"Saved outputs to {sample_output_dir}")

    if len(sample_payloads) > 1:
        overall_values = [payload["metrics"]["overall_pearson"] for payload in sample_payloads]
        gene_values = [payload["metrics"]["mean_gene_pearson"] for payload in sample_payloads]
        spot_values = [payload["metrics"]["mean_spot_pearson"] for payload in sample_payloads]
        coverage_values = [payload["spot_coverage"] for payload in sample_payloads]

        summary = {
            "num_samples": int(len(sample_payloads)),
            "samples": [payload["sample_id"] for payload in sample_payloads],
            "checkpoint_path": os.path.abspath(args.checkpoint_path),
            "aggregation": args.aggregation,
            "target_space": "log1p_counts",
            "count_semantics_verified": all(payload["count_semantics_verified"] for payload in sample_payloads),
            "average_overall_pearson": float(np.mean(overall_values)),
            "std_overall_pearson": float(np.std(overall_values)),
            "average_mean_gene_pearson": float(np.mean(gene_values)),
            "std_mean_gene_pearson": float(np.std(gene_values)),
            "average_mean_spot_pearson": float(np.mean(spot_values)),
            "std_mean_spot_pearson": float(np.std(spot_values)),
            "average_spot_coverage": float(np.mean(coverage_values)),
            "per_sample": sample_payloads,
        }
        save_json(summary, os.path.join(args.output_dir, "summary.json"))
        print(
            "Average Pearson: overall={overall:.4f}, mean_gene={gene:.4f}, mean_spot={spot:.4f}".format(
                overall=summary["average_overall_pearson"],
                gene=summary["average_mean_gene_pearson"],
                spot=summary["average_mean_spot_pearson"],
            )
        )
        print(f"Saved summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
