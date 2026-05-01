#!/usr/bin/env python3
"""Run 5-fold joint BRCA experiments into a single parent directory."""

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
TRAIN_SCRIPT = "/data/yujk/GLIP/train_joint_brca_naive.py"


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
    parser = argparse.ArgumentParser(description="Run all 5 joint folds into one parent directory")
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--visium-fold-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--xenium-sample-id", default="NCBI784")
    parser.add_argument("--xenium-reference-visium-sample-id", default="SPA124")
    parser.add_argument("--shared-gene-file", default=DEFAULT_GENE_FILE)
    parser.add_argument("--xenium-num-position-folds", type=int, default=5)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--pretrained", default="false")
    parser.add_argument("--image-encoder-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024)
    parser.add_argument("--eval-bank-mode", default="target", choices=["target", "joint"])
    parser.add_argument("--module-platform-token", default="false")
    parser.add_argument("--module-shared-private", default="false")
    parser.add_argument("--shared-private-dim", type=int, default=256)
    parser.add_argument("--private-dim", type=int, default=64)
    parser.add_argument("--private-gate", type=float, default=0.25)
    parser.add_argument("--shared-align-weight", type=float, default=0.05)
    parser.add_argument("--orth-weight", type=float, default=0.01)
    parser.add_argument("--module-image-ot", default="false")
    parser.add_argument("--module-gene-ot", default="false")
    parser.add_argument("--ot-transport", choices=["ot", "uot"], default="ot")
    parser.add_argument("--ot-image-weight", type=float, default=0.05)
    parser.add_argument("--ot-gene-weight", type=float, default=0.05)
    parser.add_argument("--uot-marginal-weight", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.parent_dir, exist_ok=True)
    visium_metrics: List[Dict] = []
    xenium_metrics: List[Dict] = []

    for fold_index in range(args.xenium_num_position_folds):
        run_dir = str(Path(args.parent_dir) / f"fold_{fold_index:02d}")
        print(f"\n=== Running Joint fold {fold_index} ===", flush=True)
        cmd = [
            sys.executable,
            TRAIN_SCRIPT,
            "--run-dir",
            run_dir,
            "--visium-fold-manifest",
            args.visium_fold_manifest,
            "--visium-fold-index",
            str(fold_index),
            "--xenium-sample-id",
            args.xenium_sample_id,
            "--xenium-reference-visium-sample-id",
            args.xenium_reference_visium_sample_id,
            "--shared-gene-file",
            args.shared_gene_file,
            "--xenium-test-fold",
            str(fold_index),
            "--xenium-num-position-folds",
            str(args.xenium_num_position_folds),
            "--model",
            args.model,
            "--pretrained",
            str(args.pretrained),
            "--image-encoder-checkpoint",
            args.image_encoder_checkpoint,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--top-k",
            str(args.top_k),
            "--retrieval-chunk-size",
            str(args.retrieval_chunk_size),
            "--eval-bank-mode",
            args.eval_bank_mode,
            "--module-platform-token",
            str(args.module_platform_token),
            "--module-shared-private",
            str(args.module_shared_private),
            "--shared-private-dim",
            str(args.shared_private_dim),
            "--private-dim",
            str(args.private_dim),
            "--private-gate",
            str(args.private_gate),
            "--shared-align-weight",
            str(args.shared_align_weight),
            "--orth-weight",
            str(args.orth_weight),
            "--module-image-ot",
            str(args.module_image_ot),
            "--module-gene-ot",
            str(args.module_gene_ot),
            "--ot-transport",
            args.ot_transport,
            "--ot-image-weight",
            str(args.ot_image_weight),
            "--ot-gene-weight",
            str(args.ot_gene_weight),
            "--uot-marginal-weight",
            str(args.uot_marginal_weight),
        ]
        subprocess.run(cmd, check=True)

        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vis_metric = dict(payload["visium_test_metrics"])
        xen_metric = dict(payload["xenium_test_metrics"])
        vis_metric["fold_idx"] = int(fold_index)
        xen_metric["fold_idx"] = int(fold_index)
        visium_metrics.append(vis_metric)
        xenium_metrics.append(xen_metric)

    summary = {
        "visium_test_summary": summarize(visium_metrics),
        "xenium_test_summary": summarize(xenium_metrics),
    }
    with open(Path(args.parent_dir) / "cv_summary_launcher.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nSaved launcher summary to", Path(args.parent_dir) / "cv_summary_launcher.json")


if __name__ == "__main__":
    main()
