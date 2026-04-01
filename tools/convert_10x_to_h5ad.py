#!/usr/bin/env python3

import argparse
import os
from typing import List, Tuple

import anndata as ad
import pandas as pd
from scipy import io, sparse


def read_barcodes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]


def read_genes(path: str) -> Tuple[List[str], List[str]]:
    gene_ids: List[str] = []
    gene_symbols: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            gene_id = parts[0]
            gene_symbol = parts[1] if len(parts) > 1 else parts[0]
            gene_ids.append(gene_id)
            gene_symbols.append(gene_symbol)
    return gene_ids, gene_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw 10x matrix directory to h5ad for GLIP scRNA loading")
    parser.add_argument("--input-dir", required=True, help="Directory containing matrix.mtx, barcodes.tsv, and genes.tsv")
    parser.add_argument("--output-path", required=True, help="Output h5ad path")
    parser.add_argument(
        "--var-index",
        choices=["gene_symbol", "gene_id"],
        default="gene_symbol",
        help="Column stored in var/_index; GLIP should use gene_symbol for Xenium symbol matching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matrix_path = os.path.join(args.input_dir, "matrix.mtx")
    barcodes_path = os.path.join(args.input_dir, "barcodes.tsv")
    genes_path = os.path.join(args.input_dir, "genes.tsv")

    for path in (matrix_path, barcodes_path, genes_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required 10x file: {path}")

    barcodes = read_barcodes(barcodes_path)
    gene_ids, gene_symbols = read_genes(genes_path)

    matrix = io.mmread(matrix_path)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    matrix = matrix.tocsr()

    expected_gene_cell_shape = (len(gene_ids), len(barcodes))
    expected_cell_gene_shape = (len(barcodes), len(gene_ids))
    if matrix.shape == expected_gene_cell_shape:
        matrix = matrix.transpose().tocsr()
    elif matrix.shape != expected_cell_gene_shape:
        raise ValueError(
            f"Unexpected matrix shape {matrix.shape}; expected {expected_gene_cell_shape} or {expected_cell_gene_shape}"
        )

    var_index_values = gene_symbols if args.var_index == "gene_symbol" else gene_ids
    obs = pd.DataFrame(index=pd.Index(barcodes, name="cell_id"))
    var = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": gene_symbols,
        },
        index=pd.Index(var_index_values, name=args.var_index),
    )

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    adata.write_h5ad(args.output_path)

    print(f"Saved {args.output_path}")
    print(f"shape={adata.shape} nnz={int(adata.X.nnz)} var_index={args.var.index.name}")


if __name__ == "__main__":
    main()
