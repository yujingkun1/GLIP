# Experiment 2 execution touchpoints

## Verified external dataset root
- `/data/yujk/hovernet2feature/HEST/hest_data_experiment2_batch_hestenv_dbg`

## Verified formal external IDs
- NCBI681
- NCBI682
- NCBI683
- NCBI684

## Code touchpoints
- `glip/visium/dataset.py`: HEST-backed sample discovery and loading
- `train_visium.py`: LOO Visium training/evaluation entrypoint
- `train_joint_brca_naive.py`: joint training + retrieval evaluation entrypoint

## Intended next execution use
1. Use current core training bank on BRCA Breast IDC samples
2. Reuse HEST-backed dataset loading for the external Lymph node IDC set
3. Add or script an external evaluation step without mixing the external set into training
