#!/usr/bin/env python3
"""Run fixed-manifest 5-fold Visium training into a single parent directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_MANIFEST = "/data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json"
DEFAULT_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt"
DEFAULT_CHECKPOINT = "/data/yujk/BLEEP/checkpoints/resnet50_a1_0-14fe96d1.pth"
TRAIN_SCRIPT = "/data/yujk/GLIP/train_visium.py"


def load_folds(manifest_path: str) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    folds = payload.get("folds") if isinstance(payload, dict) else payload
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Invalid fold manifest: {manifest_path}")
    return sorted(folds, key=lambda item: int(item["fold_index"]))


def summarize(metrics_list: List[Dict]) -> Dict:
    overall = [float(item["overall_pearson"]) for item in metrics_list]
    mean_gene = [float(item["mean_gene_pearson"]) for item in metrics_list]
    return {
        "num_folds": len(metrics_list),
        "average_overall_pearson": sum(overall) / len(overall),
        "average_mean_gene_pearson": sum(mean_gene) / len(mean_gene),
        "folds": metrics_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 5 Visium folds into one parent directory")
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--fold-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--gene-file", default=DEFAULT_GENE_FILE)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--pretrained", default="false")
    parser.add_argument("--image-encoder-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.parent_dir, exist_ok=True)
    folds = load_folds(args.fold_manifest)
    fold_metrics: List[Dict] = []

    for fold in folds:
        fold_index = int(fold["fold_index"])
        print(f"\n=== Running Visium fold {fold_index} ===", flush=True)
        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--exp_name",
            args.parent_dir,
            "--cv_mode",
            "fixed_manifest",
            "--fold_manifest",
            args.fold_manifest,
            "--fold_index",
            str(fold_index),
            "--gene_file",
            args.gene_file,
            "--model",
            args.model,
            "--pretrained",
            str(args.pretrained),
            "--image_encoder_checkpoint",
            args.image_encoder_checkpoint,
            "--batch_size",
            str(args.batch_size),
            "--max_epochs",
            str(args.max_epochs),
            "--num_workers",
            str(args.num_workers),
            "--top_k",
            str(args.top_k),
            "--retrieval_chunk_size",
            str(args.retrieval_chunk_size),
            "--device_id",
            str(args.device_id),
        ]
        subprocess.run(cmd, check=True)

        metrics_path = Path(args.parent_dir) / f"fold_{fold_index:02d}" / "pearson_metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        fold_metrics.append(metrics)

    summary = summarize(fold_metrics)
    with open(Path(args.parent_dir) / "cv_summary_launcher.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nSaved launcher summary to", Path(args.parent_dir) / "cv_summary_launcher.json")


if __name__ == "__main__":
    main()
