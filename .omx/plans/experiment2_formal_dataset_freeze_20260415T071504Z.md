# Experiment 2 formal dataset freeze

## Formal purpose
Evaluate generalization within the same BRCA line (HEST-labeled as IDC) across different tissues.

## Core training side
- Visium core: `SPA119-154` (`Breast`, `IDC`)
- Xenium core: `NCBI784` (`Breast`, `IDC`)

## Formal unseen-data benchmark
Downloaded official HEST Visium samples:
- `NCBI681`
- `NCBI682`
- `NCBI683`
- `NCBI684`

These satisfy:
- same cancer line: `IDC` (BRCA line)
- different tissue/organ: `Lymph node` rather than `Breast`

## Local download root
- `/data/yujk/hovernet2feature/HEST/hest_data_experiment2_batch_hestenv_dbg`

## Fresh verification evidence
For each of `NCBI681-684`, the following core files exist:
- `st/{id}.h5ad`
- `metadata/{id}.json`
- `wsis/{id}.tif`
- `patches/{id}.h5`
- `spatial_plots/{id}_spatial_plots.png`

## Status
Experiment 2 formal external-test dataset is now grounded locally and can be used in the next execution step.
