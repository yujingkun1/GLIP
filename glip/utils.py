"""Utility helpers for preprocessing, training, and evaluation."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


class AvgMeter:
    def __init__(self, name: str = "Metric") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.sum += float(value) * int(count)
        self.count += int(count)
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def save_json(payload: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def compute_pearson_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    entity_label: str = "cell",
) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    overall_pearson = safe_pearson(predictions.reshape(-1), targets.reshape(-1))
    gene_pearsons = [
        safe_pearson(predictions[:, gene_idx], targets[:, gene_idx])
        for gene_idx in range(predictions.shape[1])
    ]
    entity_pearsons = [
        safe_pearson(predictions[entity_idx], targets[entity_idx])
        for entity_idx in range(predictions.shape[0])
    ]

    gene_pearsons_np = np.asarray(gene_pearsons, dtype=np.float32)
    entity_pearsons_np = np.asarray(entity_pearsons, dtype=np.float32)

    return {
        "overall_pearson": float(overall_pearson),
        "mean_gene_pearson": float(gene_pearsons_np.mean()) if gene_pearsons_np.size else 0.0,
        "std_gene_pearson": float(gene_pearsons_np.std()) if gene_pearsons_np.size else 0.0,
        f"mean_{entity_label}_pearson": float(entity_pearsons_np.mean()) if entity_pearsons_np.size else 0.0,
        f"std_{entity_label}_pearson": float(entity_pearsons_np.std()) if entity_pearsons_np.size else 0.0,
        "num_genes": int(predictions.shape[1]),
        f"num_{entity_label}s": int(predictions.shape[0]),
    }


def assign_position_folds(x_coords: Iterable[float], num_folds: int) -> Tuple[np.ndarray, List[float]]:
    x_coords = np.asarray(list(x_coords), dtype=np.float64)
    if x_coords.ndim != 1:
        raise ValueError("x_coords must be a 1D array-like")
    if x_coords.size == 0:
        return np.zeros((0,), dtype=np.int64), []
    if num_folds <= 1:
        return np.zeros_like(x_coords, dtype=np.int64), [float(x_coords.min()), float(x_coords.max())]

    x_min = float(x_coords.min())
    x_max = float(x_coords.max())
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x_coords, dtype=np.int64), [x_min] * (num_folds + 1)

    edges = np.linspace(x_min, x_max, num_folds + 1, dtype=np.float64)
    fold_ids = np.digitize(x_coords, edges[1:-1], right=False).astype(np.int64)
    return fold_ids, edges.tolist()


def summarize_split(fold_ids: np.ndarray) -> Dict[str, int]:
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    unique_ids, counts = np.unique(fold_ids, return_counts=True)
    return {f"fold_{int(fold_id)}": int(count) for fold_id, count in zip(unique_ids, counts)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sample_indices(total_size: int, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or total_size <= max_items:
        return np.arange(total_size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_size, size=max_items, replace=False).astype(np.int64))
