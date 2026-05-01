#!/usr/bin/env python3
"""Run 5 Xenium pseudo-spot folds into a single parent directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_GENE_FILE = "/data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt"
DEFAULT_CHECKPOINT = "/data/yujk/BLEEP/checkpoints/resnet50_a1_0-14fe96d1.pth"
TRAIN_SCRIPT = "/data/yujk/GLIP/train_xenium_pseudospot.py"


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
    parser = argparse.ArgumentParser(description="Run all 5 Xenium folds into one parent directory")
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--sample-id", default="NCBI784")
    parser.add_argument("--reference-visium-sample-id", default="SPA124")
    parser.add_argument("--gene-file", default=DEFAULT_GENE_FILE)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--pretrained", default="false")
    parser.add_argument("--image-encoder-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-position-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.parent_dir, exist_ok=True)
    fold_metrics: List[Dict] = []

    for fold_index in range(args.num_position_folds):
        run_dir = str(Path(args.parent_dir) / f"fold_{fold_index:02d}")
        print(f"\n=== Running Xenium fold {fold_index} ===", flush=True)
        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--run-dir",
            run_dir,
            "--sample-id",
            args.sample_id,
            "--reference-visium-sample-id",
            args.reference_visium_sample_id,
            "--gene-file",
            args.gene_file,
            "--model",
            args.model,
            "--pretrained",
            str(args.pretrained),
            "--image-encoder-checkpoint",
            args.image_encoder_checkpoint,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--test-fold",
            str(fold_index),
            "--num-position-folds",
            str(args.num_position_folds),
            "--top-k",
            str(args.top_k),
            "--retrieval-chunk-size",
            str(args.retrieval_chunk_size),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)

        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = dict(payload["final_test_metrics"])
        metrics["fold_idx"] = int(fold_index)
        fold_metrics.append(metrics)

    summary = summarize(fold_metrics)
    with open(Path(args.parent_dir) / "cv_summary_launcher.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nSaved launcher summary to", Path(args.parent_dir) / "cv_summary_launcher.json")


if __name__ == "__main__":
    main()
