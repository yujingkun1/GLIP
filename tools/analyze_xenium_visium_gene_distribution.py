#!/usr/bin/env python3
"""Compare Xenium and Visium gene-expression distributions on overlapping genes."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from glip.data import is_control_feature
from glip.utils import parse_bool, sample_indices, save_json, seed_everything


@dataclass
class SampleMatrix:
    sample_id: str
    technology: str
    path: str
    row_indices: np.ndarray
    obs_df: pd.DataFrame
    var_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gene distributions across Xenium and Visium samples")
    parser.add_argument(
        "--xenium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium",
        help="HEST Xenium root directory",
    )
    parser.add_argument(
        "--visium-hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data",
        help="HEST Visium/ST root directory",
    )
    parser.add_argument("--xenium-sample-id", default="NCBI784", help="Xenium sample id")
    parser.add_argument(
        "--visium-sample-ids",
        default="SPA124",
        help="Comma-separated Visium sample ids to compare against the Xenium sample",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/runs/gene_gap_analysis",
        help="Directory for gene distribution outputs",
    )
    parser.add_argument(
        "--in-tissue-only",
        default="true",
        help="Whether to keep only in-tissue observations when the flag exists",
    )
    parser.add_argument(
        "--max-spots-per-sample",
        type=int,
        default=0,
        help="Optional cap on rows per sample for balanced UMAP, 0 means all rows",
    )
    parser.add_argument(
        "--gene-top-k",
        type=int,
        default=12,
        help="Number of top shifted genes to visualize with per-gene distributions",
    )
    parser.add_argument(
        "--gene-min-detect-fraction",
        type=float,
        default=0.05,
        help="Only rank genes with at least this pooled detection fraction",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def parse_sample_id_list(raw_value: str) -> List[str]:
    parsed = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one Visium sample id must be provided.")
    return parsed


def normalize_log1p_cpm(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    total = counts.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(counts, dtype=np.float32)
    valid = total[:, 0] > 0
    if valid.any():
        normalized[valid] = np.log1p(counts[valid] / total[valid] * 1e4)
    return normalized


def compute_embedding_separation_metrics(features: np.ndarray, labels: Sequence[str]) -> Dict[str, float | None]:
    encoded = np.asarray(pd.Categorical(labels).codes, dtype=np.int64)
    unique_codes = np.unique(encoded)
    metrics: Dict[str, float | None] = {
        "silhouette": None,
        "same_label_knn_fraction_k15": None,
    }
    if features.shape[0] < 3 or unique_codes.size < 2:
        return metrics

    if features.shape[0] > unique_codes.size:
        metrics["silhouette"] = float(silhouette_score(features, encoded))

    k = min(15, features.shape[0] - 1)
    if k >= 1:
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(features)
        neighbor_indices = knn.kneighbors(return_distance=False)[:, 1:]
        metrics["same_label_knn_fraction_k15"] = float(np.mean(encoded[:, None] == encoded[neighbor_indices]))
    return metrics


def run_umap(
    *,
    features: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    pca_dim = max(2, min(32, standardized.shape[0] - 1, standardized.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca_features = pca.fit_transform(standardized)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(int(n_neighbors), pca_features.shape[0] - 1)),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=seed,
    )
    embedding = reducer.fit_transform(pca_features)
    return embedding, pca_features, {
        "pca_components": int(pca_dim),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def save_umap_scatter(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8.5, 7.0))
    ax = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        s=38,
        alpha=0.82,
        linewidth=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def prepare_sample_matrix(
    *,
    hest_data_dir: str,
    sample_id: str,
    technology: str,
    in_tissue_only: bool,
    max_rows: int,
    seed: int,
) -> SampleMatrix:
    path = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing h5ad file: {path}")

    adata = ad.read_h5ad(path, backed="r")
    try:
        keep_mask = np.ones(int(adata.n_obs), dtype=bool)
        if in_tissue_only and "in_tissue" in adata.obs.columns:
            keep_mask &= adata.obs["in_tissue"].to_numpy().astype(np.int64, copy=False) > 0
        row_indices = np.flatnonzero(keep_mask)
        if max_rows and max_rows > 0 and row_indices.size > max_rows:
            chosen = sample_indices(row_indices.size, max_rows, seed)
            row_indices = row_indices[chosen]
        obs_df = adata.obs.iloc[row_indices].copy()
        obs_df["obs_name"] = [str(adata.obs_names[idx]) for idx in row_indices]
        var_names = [str(gene) for gene in adata.var_names]
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

    obs_df["sample_id"] = sample_id
    obs_df["technology"] = technology
    return SampleMatrix(
        sample_id=sample_id,
        technology=technology,
        path=path,
        row_indices=row_indices.astype(np.int64, copy=False),
        obs_df=obs_df.reset_index(drop=True),
        var_names=var_names,
    )


def load_counts_for_genes(sample: SampleMatrix, overlap_genes: Sequence[str]) -> np.ndarray:
    adata = ad.read_h5ad(sample.path, backed="r")
    try:
        gene_indexer = adata.var_names.get_indexer(list(overlap_genes))
        if np.any(gene_indexer < 0):
            missing = [overlap_genes[idx] for idx, val in enumerate(gene_indexer) if val < 0][:10]
            raise RuntimeError(f"Sample {sample.sample_id} is missing overlap genes unexpectedly: {missing}")
        matrix = adata.X[sample.row_indices][:, gene_indexer]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def build_overlap_gene_list(samples: Sequence[SampleMatrix]) -> List[str]:
    overlap = set(samples[0].var_names)
    for sample in samples[1:]:
        overlap &= set(sample.var_names)
    overlap_genes = [gene for gene in samples[0].var_names if gene in overlap and not is_control_feature(gene)]
    if not overlap_genes:
        raise RuntimeError("No overlapping non-control genes found across selected samples.")
    return overlap_genes


def save_qc_distributions(metadata_df: pd.DataFrame, output_dir: str) -> None:
    qc_columns = [col for col in ["total_counts", "n_counts", "n_genes_by_counts"] if col in metadata_df.columns]
    if not qc_columns:
        return
    fig, axes = plt.subplots(1, len(qc_columns), figsize=(6.0 * len(qc_columns), 4.8))
    axes = np.asarray(axes, dtype=object).reshape(1, len(qc_columns))
    for ax, column in zip(axes.flat, qc_columns):
        sns.kdeplot(
            data=metadata_df,
            x=column,
            hue="sample_id",
            common_norm=False,
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(column)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spot_qc_distribution.png"), dpi=220)
    plt.close()


def save_gene_scatter(
    *,
    gene_summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    output_path: str,
    top_label_count: int,
) -> None:
    max_axis = float(max(gene_summary_df[x_col].max(), gene_summary_df[y_col].max()))
    plt.figure(figsize=(7.2, 7.0))
    ax = sns.scatterplot(data=gene_summary_df, x=x_col, y=y_col, s=24, alpha=0.7, linewidth=0.0)
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="black", linewidth=1.2)
    for row in gene_summary_df.head(top_label_count).itertuples(index=False):
        ax.text(getattr(row, x_col), getattr(row, y_col), getattr(row, label_col), fontsize=9)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_top_gene_distributions(
    *,
    expression_df: pd.DataFrame,
    gene_summary_df: pd.DataFrame,
    output_dir: str,
    gene_top_k: int,
) -> None:
    top_genes = gene_summary_df["gene"].head(gene_top_k).tolist()
    if not top_genes:
        return

    cols = 3
    rows = int(np.ceil(len(top_genes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 4.2))
    axes = np.asarray(axes, dtype=object).reshape(rows, cols)

    for ax, gene_name in zip(axes.flat, top_genes):
        subset = expression_df.loc[expression_df["gene"] == gene_name]
        sns.histplot(
            data=subset,
            x="expression",
            hue="technology",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            bins=32,
            ax=ax,
        )
        ax.set_title(gene_name)
        ax.set_xlabel("log1p(CPM)")
        ax.set_ylabel("Density")
    for ax in axes.flat[len(top_genes) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_gene_distribution.png"), dpi=220)
    plt.close()


def build_gene_summary(
    *,
    overlap_genes: Sequence[str],
    counts_by_sample: Dict[str, np.ndarray],
    normalized_by_sample: Dict[str, np.ndarray],
    samples: Sequence[SampleMatrix],
    min_detect_fraction: float,
) -> pd.DataFrame:
    technology_to_sample_ids: Dict[str, List[str]] = {}
    for sample in samples:
        technology_to_sample_ids.setdefault(sample.technology, []).append(sample.sample_id)

    stats: Dict[str, np.ndarray] = {}
    for technology, sample_ids in technology_to_sample_ids.items():
        counts = np.concatenate([counts_by_sample[sample_id] for sample_id in sample_ids], axis=0)
        normalized = np.concatenate([normalized_by_sample[sample_id] for sample_id in sample_ids], axis=0)
        stats[f"{technology.lower()}_mean"] = normalized.mean(axis=0)
        stats[f"{technology.lower()}_detect"] = (counts > 0).mean(axis=0)

    summary = pd.DataFrame({"gene": list(overlap_genes)})
    for key, values in stats.items():
        summary[key] = values.astype(np.float32)

    tech_names = sorted({sample.technology for sample in samples})
    if len(tech_names) == 2:
        a = tech_names[0].lower()
        b = tech_names[1].lower()
        summary["abs_mean_diff"] = np.abs(summary[f"{a}_mean"] - summary[f"{b}_mean"])
        summary["abs_detect_diff"] = np.abs(summary[f"{a}_detect"] - summary[f"{b}_detect"])
    else:
        mean_cols = [f"{name.lower()}_mean" for name in tech_names]
        detect_cols = [f"{name.lower()}_detect" for name in tech_names]
        summary["abs_mean_diff"] = summary[mean_cols].max(axis=1) - summary[mean_cols].min(axis=1)
        summary["abs_detect_diff"] = summary[detect_cols].max(axis=1) - summary[detect_cols].min(axis=1)

    pooled_counts = np.concatenate(list(counts_by_sample.values()), axis=0)
    summary["overall_detect"] = (pooled_counts > 0).mean(axis=0)
    summary = summary.loc[summary["overall_detect"] >= float(min_detect_fraction)].copy()
    summary = summary.sort_values(["abs_mean_diff", "abs_detect_diff"], ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    args = parse_args()
    args.visium_sample_ids = parse_sample_id_list(args.visium_sample_ids)
    args.in_tissue_only = parse_bool(args.in_tissue_only)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    sns.set_theme(style="whitegrid", context="talk")

    samples: List[SampleMatrix] = []
    samples.append(
        prepare_sample_matrix(
            hest_data_dir=args.xenium_hest_data_dir,
            sample_id=args.xenium_sample_id,
            technology="Xenium",
            in_tissue_only=args.in_tissue_only,
            max_rows=args.max_spots_per_sample,
            seed=args.seed,
        )
    )
    for idx, sample_id in enumerate(args.visium_sample_ids):
        samples.append(
            prepare_sample_matrix(
                hest_data_dir=args.visium_hest_data_dir,
                sample_id=sample_id,
                technology="Visium",
                in_tissue_only=args.in_tissue_only,
                max_rows=args.max_spots_per_sample,
                seed=args.seed + 1000 + idx,
            )
        )

    overlap_genes = build_overlap_gene_list(samples)
    counts_by_sample: Dict[str, np.ndarray] = {}
    normalized_by_sample: Dict[str, np.ndarray] = {}
    metadata_frames: List[pd.DataFrame] = []
    combined_expression: List[np.ndarray] = []

    for sample in samples:
        counts = load_counts_for_genes(sample, overlap_genes)
        normalized = normalize_log1p_cpm(counts)
        counts_by_sample[sample.sample_id] = counts
        normalized_by_sample[sample.sample_id] = normalized
        sample_meta = sample.obs_df.copy()
        sample_meta["row_count_sum_overlap"] = counts.sum(axis=1).astype(np.float32)
        metadata_frames.append(sample_meta)
        combined_expression.append(normalized)

    metadata_df = pd.concat(metadata_frames, axis=0, ignore_index=True)
    combined_expression_matrix = np.concatenate(combined_expression, axis=0)

    gene_summary_df = build_gene_summary(
        overlap_genes=overlap_genes,
        counts_by_sample=counts_by_sample,
        normalized_by_sample=normalized_by_sample,
        samples=samples,
        min_detect_fraction=args.gene_min_detect_fraction,
    )
    gene_summary_df.to_csv(os.path.join(args.output_dir, "gene_summary.csv"), index=False)

    save_qc_distributions(metadata_df, args.output_dir)

    expression_umap, expression_pca, expression_meta = run_umap(
        features=combined_expression_matrix,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    metadata_df["expression_umap_1"] = expression_umap[:, 0]
    metadata_df["expression_umap_2"] = expression_umap[:, 1]
    metadata_df.to_csv(os.path.join(args.output_dir, "spot_expression_with_umap.csv"), index=False)

    save_umap_scatter(
        df=metadata_df,
        x_col="expression_umap_1",
        y_col="expression_umap_2",
        hue_col="technology",
        style_col="sample_id",
        title="Expression UMAP by Technology",
        output_path=os.path.join(args.output_dir, "expression_umap_by_technology.png"),
    )
    save_umap_scatter(
        df=metadata_df,
        x_col="expression_umap_1",
        y_col="expression_umap_2",
        hue_col="sample_id",
        style_col="technology",
        title="Expression UMAP by Sample",
        output_path=os.path.join(args.output_dir, "expression_umap_by_sample.png"),
    )

    mean_cols = [col for col in gene_summary_df.columns if col.endswith("_mean")]
    detect_cols = [col for col in gene_summary_df.columns if col.endswith("_detect")]
    if len(mean_cols) >= 2:
        save_gene_scatter(
            gene_summary_df=gene_summary_df,
            x_col=mean_cols[0],
            y_col=mean_cols[1],
            label_col="gene",
            title="Per-Gene Mean Expression Comparison",
            output_path=os.path.join(args.output_dir, "gene_mean_scatter.png"),
            top_label_count=min(12, len(gene_summary_df)),
        )
    if len(detect_cols) >= 2:
        save_gene_scatter(
            gene_summary_df=gene_summary_df,
            x_col=detect_cols[0],
            y_col=detect_cols[1],
            label_col="gene",
            title="Per-Gene Detection Rate Comparison",
            output_path=os.path.join(args.output_dir, "gene_detection_scatter.png"),
            top_label_count=min(12, len(gene_summary_df)),
        )

    top_genes = gene_summary_df["gene"].head(args.gene_top_k).tolist()
    if top_genes:
        long_frames: List[pd.DataFrame] = []
        for sample in samples:
            sample_matrix = normalized_by_sample[sample.sample_id]
            gene_positions = {gene: idx for idx, gene in enumerate(overlap_genes)}
            for gene_name in top_genes:
                long_frames.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample.sample_id,
                            "technology": sample.technology,
                            "gene": gene_name,
                            "expression": sample_matrix[:, gene_positions[gene_name]],
                        }
                    )
                )
        expression_df = pd.concat(long_frames, axis=0, ignore_index=True)
        expression_df.to_csv(os.path.join(args.output_dir, "top_gene_expression_long.csv"), index=False)
        save_top_gene_distributions(
            expression_df=expression_df,
            gene_summary_df=gene_summary_df,
            output_dir=args.output_dir,
            gene_top_k=args.gene_top_k,
        )

    summary_payload = {
        "xenium_sample_id": args.xenium_sample_id,
        "visium_sample_ids": args.visium_sample_ids,
        "num_overlap_genes_all": int(len(overlap_genes)),
        "num_overlap_genes_after_filter": int(len(gene_summary_df)),
        "sampled_rows_per_sample": {sample.sample_id: int(sample.row_indices.shape[0]) for sample in samples},
        "expression_feature_space": {
            **expression_meta,
            "technology_separation": compute_embedding_separation_metrics(
                expression_pca,
                metadata_df["technology"].tolist(),
            ),
            "sample_separation": compute_embedding_separation_metrics(
                expression_pca,
                metadata_df["sample_id"].tolist(),
            ),
        },
        "top_shifted_genes": top_genes,
    }
    save_json(summary_payload, os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
