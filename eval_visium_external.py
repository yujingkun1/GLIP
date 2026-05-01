import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from glip.visium.dataset import CLIPDataset
from train_visium import (
    UNI_MODEL_NAME,
    build_model,
    collect_image_queries,
    collect_spot_bank,
    compute_pearson_metrics,
    configure_hf_hub,
    create_loader,
    parse_bool,
    predict_expression_from_retrieval,
    resolve_image_model_name,
)

DEFAULT_CORE_SAMPLE_IDS = [
    "SPA154", "SPA153", "SPA152", "SPA151", "SPA150", "SPA149",
    "SPA148", "SPA147", "SPA146", "SPA145", "SPA144", "SPA143",
    "SPA142", "SPA141", "SPA140", "SPA139", "SPA138", "SPA137",
    "SPA136", "SPA135", "SPA134", "SPA133", "SPA132", "SPA131",
    "SPA130", "SPA129", "SPA128", "SPA127", "SPA126", "SPA125",
    "SPA124", "SPA123", "SPA122", "SPA121", "SPA120", "SPA119",
]


def parse_sample_ids(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [sample_id.strip() for sample_id in raw.split(",") if sample_id.strip()]
    return [str(sample_id).strip() for sample_id in raw if str(sample_id).strip()]


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_gene_file(genes, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for gene in genes:
            handle.write(f"{gene}\n")


def summarize_per_sample(predictions, targets, sample_ids):
    rows = []
    sample_ids = np.asarray(sample_ids)
    for sample_id in sorted(set(sample_ids.tolist())):
        mask = sample_ids == sample_id
        sample_metrics = compute_pearson_metrics(predictions[mask], targets[mask])
        sample_metrics["sample_id"] = sample_id
        rows.append(sample_metrics)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Visium model on external HEST Visium samples")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--train_hest_data_dir", default="/data/yujk/hovernet2feature/HEST/hest_data")
    parser.add_argument("--train_sample_ids", default="")
    parser.add_argument("--external_hest_data_dir", required=True)
    parser.add_argument("--external_sample_ids", required=True)
    parser.add_argument("--gene_file", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--retrieval_chunk_size", type=int, default=1024)
    parser.add_argument("--pretrained", default="true")
    parser.add_argument("--image_encoder_checkpoint", default="/data/yujk/UNI2-h/pytorch_model.bin")
    parser.add_argument("--hf_endpoint", default="")
    parser.add_argument("--hf_hub_download_timeout", type=int, default=0)
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=0)
    parser.add_argument("--model", default="uni")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    args.pretrained = parse_bool(args.pretrained)
    args.model = args.model.strip()
    args.resolved_model_name = resolve_image_model_name(args.model)
    args.hf_endpoint = args.hf_endpoint.strip()
    args.image_encoder_checkpoint = os.path.expanduser(args.image_encoder_checkpoint.strip()) if args.image_encoder_checkpoint else ""
    configure_hf_hub(args)

    if args.image_encoder_checkpoint and args.pretrained:
        args.pretrained = False

    train_sample_ids = parse_sample_ids(args.train_sample_ids) or list(DEFAULT_CORE_SAMPLE_IDS)
    external_sample_ids = parse_sample_ids(args.external_sample_ids)
    if not external_sample_ids:
        raise RuntimeError("At least one external sample id is required.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    train_dataset = CLIPDataset(
        hest_data_dir=args.train_hest_data_dir,
        sample_ids=train_sample_ids,
        gene_file=args.gene_file or None,
        is_train=False,
    )

    resolved_gene_file = args.gene_file
    if not resolved_gene_file:
        resolved_gene_file = os.path.join(args.run_dir, "train_gene_list.txt")
        write_gene_file(train_dataset.selected_genes, resolved_gene_file)

    external_dataset = CLIPDataset(
        hest_data_dir=args.external_hest_data_dir,
        sample_ids=external_sample_ids,
        gene_file=resolved_gene_file,
        is_train=False,
    )

    train_loader = create_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    external_loader = create_loader(external_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = build_model(args, train_dataset.num_features).to(device)
    payload = torch.load(args.checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    train_bank = collect_spot_bank(model, train_loader, device)
    external_queries = collect_image_queries(model, external_loader, device)
    predictions = predict_expression_from_retrieval(train_bank, external_queries, top_k=args.top_k, chunk_size=args.retrieval_chunk_size)
    targets = external_queries["expressions"].numpy()
    metrics = compute_pearson_metrics(predictions, targets)
    metrics["top_k"] = int(args.top_k)
    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["train_hest_data_dir"] = args.train_hest_data_dir
    metrics["external_hest_data_dir"] = args.external_hest_data_dir
    metrics["train_sample_ids"] = train_sample_ids
    metrics["external_sample_ids"] = external_sample_ids
    metrics["gene_file"] = resolved_gene_file
    metrics["retrieval_bank_size"] = int(train_bank["embeddings"].shape[0])
    metrics["external_query_count"] = int(external_queries["embeddings"].shape[0])

    per_sample = summarize_per_sample(predictions, targets, external_queries["sample_ids"])

    save_json(metrics, os.path.join(args.run_dir, "external_metrics.json"))
    save_json(per_sample, os.path.join(args.run_dir, "external_per_sample_metrics.json"))

    with open(os.path.join(args.run_dir, "external_per_sample_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({k for row in per_sample for k in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample)

    print(json.dumps(metrics, indent=2))
    print(json.dumps(per_sample, indent=2))


if __name__ == "__main__":
    main()
