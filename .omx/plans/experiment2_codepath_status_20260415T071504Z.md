# Experiment 2 codepath status

## Fresh evidence
- External HEST root is valid and discoverable.
- `discover_hest_sample_ids(...)` returns `NCBI681-684`.
- `train_visium.py` exposes `--hest_data_dir`, `--sample_ids`, `--gene_file`, and HF endpoint flags.

## Initial implication
- Internal/external HEST dataset loading is already reusable.
- The remaining question is whether current training scripts already support **train on core BRCA Breast IDC, test on external Lymph node IDC** directly, or whether a dedicated external-eval script is needed.

## Next execution need
Inspect current run artifacts/checkpoints and determine whether to:
1. reuse an existing checkpoint for external eval, or
2. implement a dedicated external evaluation entrypoint.
