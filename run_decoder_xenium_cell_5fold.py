#!/usr/bin/env python3
"""Run 5-fold decoder evaluation on Xenium single cells."""

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
EVAL_SCRIPT = "/data/yujk/GLIP/eval_decoder_on_xenium_cells.py"


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
    parser = argparse.ArgumentParser(description="Run all 5 decoder-on-Xenium-cell folds into one parent directory")
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--checkpoint-parent-dir", required=True)
    parser.add_argument("--sample-id", default="NCBI784")
    parser.add_argument("--gene-file", default=DEFAULT_GENE_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--pretrained", default="false")
    parser.add_argument("--image-encoder-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-position-folds", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-test-cells", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.parent_dir, exist_ok=True)
    fold_metrics: List[Dict] = []

    for fold_index in range(args.num_position_folds):
        run_dir = str(Path(args.parent_dir) / f"fold_{fold_index:02d}")
        checkpoint = Path(args.checkpoint_parent_dir) / f"fold_{fold_index:02d}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")

        print(f"\n=== Evaluating decoder on Xenium single-cell fold {fold_index} ===", flush=True)
        cmd = [
            sys.executable,
            EVAL_SCRIPT,
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            run_dir,
            "--sample-id",
            args.sample_id,
            "--gene-file",
            args.gene_file,
            "--pretrained",
            str(args.pretrained),
            "--image-encoder-checkpoint",
            args.image_encoder_checkpoint,
            "--test-fold",
            str(fold_index),
            "--num-position-folds",
            str(args.num_position_folds),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--max-test-cells",
            str(args.max_test_cells),
        ]
        if args.model:
            cmd.extend(["--model", args.model])
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
