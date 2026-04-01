#!/usr/bin/env python3
"""Build a strictly aligned NCBI784 single-cell cache for GLIP."""

from __future__ import annotations

import argparse

from .data import prepare_processed_dataset
from glip.utils import parse_bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NCBI784 Xenium single-cell cache")
    parser.add_argument(
        "--hest-data-dir",
        default="/data/yujk/hovernet2feature/HEST/hest_data_Xenium",
        help="HEST Xenium root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yujk/GLIP/processed",
        help="Directory used to store the processed cache",
    )
    parser.add_argument("--sample-id", default="NCBI784", help="HEST Xenium sample id")
    parser.add_argument(
        "--remove-control-features",
        default="true",
        help="Remove BLANK / NegControl / antisense features",
    )
    parser.add_argument(
        "--nucleus-only",
        default="false",
        help="Only keep transcripts with overlaps_nucleus == 1",
    )
    parser.add_argument(
        "--drop-zero-expression",
        default="true",
        help="Drop segmented cells with zero remaining transcripts",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the cache even if it already exists")
    args = parser.parse_args()

    processed_paths = prepare_processed_dataset(
        hest_data_dir=args.hest_data_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        remove_control_features=parse_bool(args.remove_control_features),
        nucleus_only=parse_bool(args.nucleus_only),
        drop_zero_expression=parse_bool(args.drop_zero_expression),
        force_rebuild=args.force_rebuild,
    )

    print(f"Counts:   {processed_paths.counts_path}")
    print(f"Metadata: {processed_paths.metadata_path}")
    print(f"Genes:    {processed_paths.genes_path}")
    print(f"Manifest: {processed_paths.manifest_path}")


if __name__ == "__main__":
    main()
