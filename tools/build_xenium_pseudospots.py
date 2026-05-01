#!/usr/bin/env python3
"""Build NCBI784 pseudo-spots and compare them with native Xenium spot-like data."""

from __future__ import annotations

import argparse
import json
import os
import sys
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

from glip.xenium.pseudospot import (
    build_pseudospot_output_dir,
    build_pseudospot_paths,
    prepare_pseudospot_dataset,
)
from glip.utils import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Xenium pseudo-spots and export diagnostics")
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
        "--reference-visium-sample-id",
        default="SPA124",
        help="Reference Visium/ST sample whose spot size defines the pseudo-spot geometry",
    )
    parser.add_argument(
        "--pseudo-output-base-dir",
        default="/data/yujk/GLIP/processed/pseudospots",
        help="Base directory where pseudo-spot folders are created",
    )
    parser.add_argument("--processed-dir", default="/data/yujk/GLIP/processed", help="Processed Xenium cell cache directory")
    parser.add_argument("--min-cells-per-spot", type=int, default=3, help="Minimum number of assigned cells per pseudo-spot")
    parser.add_argument("--grid-layout", default="auto", choices=["auto", "square", "hex"], help="Pseudo-spot grid layout")
    parser.add_argument(
        "--use-reference-inter-spot-dist",
        default="true",
        help="Whether pseudo-spot center spacing should follow the reference inter-spot distance",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild pseudo-spot cache even if files exist")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def parse_bool(value) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


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
        neighbor_indices = knn.kneighbors(features, return_distance=False)[:, 1:]
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
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(8.0, 7.0))
    ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=34, alpha=0.82, linewidth=0.0)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_qc_distributions(comparison_df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        ("total_counts", "Total Counts"),
        ("n_genes_by_counts", "Detected Genes"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.8))
    axes = np.asarray(axes, dtype=object).reshape(1, len(metrics))
    for ax, (column, title) in zip(axes.flat, metrics):
        sns.kdeplot(data=comparison_df, x=column, hue="source", common_norm=False, linewidth=2.0, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudospot_vs_xenium_qc_distribution.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(6.6, 4.8))
    ax = sns.histplot(
        data=comparison_df.loc[comparison_df["source"] == "pseudospot"],
        x="cell_count",
        bins=30,
        color="#3566B8",
    )
    ax.set_title("Pseudo-Spot Cell Count Distribution")
    ax.set_xlabel("Cells per pseudo-spot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudospot_cell_count_distribution.png"), dpi=220)
    plt.close()


def save_gene_mean_scatter(gene_summary_df: pd.DataFrame, output_path: str) -> None:
    max_axis = float(max(gene_summary_df["pseudospot_mean"].max(), gene_summary_df["xenium_mean"].max()))
    plt.figure(figsize=(7.2, 7.0))
    ax = sns.scatterplot(
        data=gene_summary_df,
        x="pseudospot_mean",
        y="xenium_mean",
        s=24,
        alpha=0.72,
        linewidth=0.0,
    )
    ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="black", linewidth=1.2)
    for row in gene_summary_df.head(12).itertuples(index=False):
        ax.text(float(row.pseudospot_mean), float(row.xenium_mean), str(row.gene), fontsize=8.8)
    ax.set_title("Pseudo-Spot vs Native Xenium Mean Expression")
    ax.set_xlabel("Pseudo-spot mean log1p(CPM)")
    ax.set_ylabel("Native Xenium mean log1p(CPM)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_native_xenium_spot_data(
    *,
    xenium_hest_data_dir: str,
    sample_id: str,
    overlap_genes: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    st_path = os.path.join(xenium_hest_data_dir, "st", f"{sample_id}.h5ad")
    adata = ad.read_h5ad(st_path, backed="r")
    try:
        gene_indexer = adata.var_names.get_indexer(list(overlap_genes))
        counts = adata.X[:, gene_indexer]
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = np.asarray(counts, dtype=np.float32)
        obs_df = adata.obs.copy()
        obs_df["obs_name"] = [str(name) for name in adata.obs_names]
        obs_df["source"] = "native_xenium"
        if "n_counts" in obs_df.columns and "total_counts" not in obs_df.columns:
            obs_df["total_counts"] = obs_df["n_counts"]
        if "n_genes_by_counts" not in obs_df.columns:
            obs_df["n_genes_by_counts"] = (counts > 0).sum(axis=1)
        if "total_counts" not in obs_df.columns:
            obs_df["total_counts"] = counts.sum(axis=1)
        obs_df["cell_count"] = np.nan
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
    return obs_df.reset_index(drop=True), counts


def main() -> None:
    args = parse_args()
    args.use_reference_inter_spot_dist = parse_bool(args.use_reference_inter_spot_dist)
    pseudo_output_dir = build_pseudospot_output_dir(
        args.pseudo_output_base_dir,
        args.xenium_sample_id,
        args.reference_visium_sample_id,
    )
    os.makedirs(pseudo_output_dir, exist_ok=True)
    seed_everything(args.seed)
    sns.set_theme(style="whitegrid", context="talk")

    paths = prepare_pseudospot_dataset(
        xenium_hest_data_dir=args.xenium_hest_data_dir,
        visium_hest_data_dir=args.visium_hest_data_dir,
        processed_dir=args.processed_dir,
        pseudo_output_dir=pseudo_output_dir,
        xenium_sample_id=args.xenium_sample_id,
        reference_sample_id=args.reference_visium_sample_id,
        min_cells_per_spot=args.min_cells_per_spot,
        grid_layout=args.grid_layout,
        use_reference_inter_spot_dist=args.use_reference_inter_spot_dist,
        force_rebuild=args.force_rebuild,
    )

    pseudo_counts = np.load(paths.counts_path)
    pseudo_metadata = pd.read_parquet(paths.metadata_path)
    with open(paths.genes_path, "r", encoding="utf-8") as handle:
        pseudo_genes = list(json.load(handle)["genes"])
    with open(paths.manifest_path, "r", encoding="utf-8") as handle:
        pseudo_manifest = json.load(handle)

    native_obs_df, native_counts = load_native_xenium_spot_data(
        xenium_hest_data_dir=args.xenium_hest_data_dir,
        sample_id=args.xenium_sample_id,
        overlap_genes=pseudo_genes,
    )

    overlap_genes = [gene for gene in pseudo_genes if gene in pseudo_genes]
    pseudo_norm = normalize_log1p_cpm(pseudo_counts)
    native_norm = normalize_log1p_cpm(native_counts)

    pseudo_comparison_df = pseudo_metadata.rename(
        columns={
            "spot_id": "obs_name",
            "transcript_count": "total_counts",
        }
    ).copy()
    pseudo_comparison_df["source"] = "pseudospot"
    comparison_df = pd.concat(
        [
            pseudo_comparison_df[["obs_name", "source", "total_counts", "n_genes_by_counts", "cell_count"]],
            native_obs_df[["obs_name", "source", "total_counts", "n_genes_by_counts", "cell_count"]],
        ],
        axis=0,
        ignore_index=True,
    )
    comparison_df.to_csv(os.path.join(pseudo_output_dir, "pseudospot_vs_xenium_qc.csv"), index=False)
    save_qc_distributions(comparison_df, pseudo_output_dir)

    combined_expression = np.concatenate([pseudo_norm, native_norm], axis=0)
    umap_embedding, pca_features, umap_meta = run_umap(
        features=combined_expression,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    expression_df = pd.DataFrame(
        {
            "source": ["pseudospot"] * pseudo_norm.shape[0] + ["native_xenium"] * native_norm.shape[0],
            "umap_1": umap_embedding[:, 0],
            "umap_2": umap_embedding[:, 1],
        }
    )
    expression_df.to_csv(os.path.join(pseudo_output_dir, "pseudospot_vs_xenium_expression_umap.csv"), index=False)
    save_umap_scatter(
        df=expression_df,
        x_col="umap_1",
        y_col="umap_2",
        hue_col="source",
        title="Pseudo-Spot vs Native Xenium Expression UMAP",
        output_path=os.path.join(pseudo_output_dir, "pseudospot_vs_xenium_expression_umap.png"),
    )

    gene_summary_df = pd.DataFrame(
        {
            "gene": pseudo_genes,
            "pseudospot_mean": pseudo_norm.mean(axis=0),
            "xenium_mean": native_norm.mean(axis=0),
        }
    )
    gene_summary_df["abs_mean_diff"] = np.abs(gene_summary_df["pseudospot_mean"] - gene_summary_df["xenium_mean"])
    gene_summary_df = gene_summary_df.sort_values("abs_mean_diff", ascending=False).reset_index(drop=True)
    gene_summary_df.to_csv(os.path.join(pseudo_output_dir, "pseudospot_vs_xenium_gene_summary.csv"), index=False)
    save_gene_mean_scatter(
        gene_summary_df=gene_summary_df,
        output_path=os.path.join(pseudo_output_dir, "pseudospot_vs_xenium_gene_mean_scatter.png"),
    )

    summary_payload = {
        "pseudospot_dir": os.path.abspath(pseudo_output_dir),
        "xenium_sample_id": args.xenium_sample_id,
        "reference_visium_sample_id": args.reference_visium_sample_id,
        "num_pseudospots": int(pseudo_counts.shape[0]),
        "num_native_xenium_spots": int(native_counts.shape[0]),
        "num_genes": int(len(pseudo_genes)),
        "pseudospot_manifest": pseudo_manifest,
        "expression_feature_space": {
            **umap_meta,
            "source_separation": compute_embedding_separation_metrics(
                pca_features,
                expression_df["source"].tolist(),
            ),
        },
        "top_shifted_genes": gene_summary_df["gene"].head(12).tolist(),
    }
    save_json(summary_payload, os.path.join(pseudo_output_dir, "pseudospot_summary.json"))


if __name__ == "__main__":
    main()
