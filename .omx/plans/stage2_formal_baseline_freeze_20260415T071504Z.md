# Stage 2 formal baseline freeze

## Source run
- `/data/yujk/GLIP/runs_joint/brca_stage2_naive_joint_targetbank_full_live`

## Formal interpretation
This is the current best full-run Stage 2 artifact for the naive joint baseline with target-bank evaluation.
It becomes the formal comparison anchor for the next single-module addition.

## Frozen configuration
- heldout Visium sample: `SPA119`
- core Visium scope: `SPA119-154`
- core Xenium scope: `NCBI784`
- shared gene file: `configs/brca_shared_genes_ncbi784_ref_spa124_313.txt`
- eval bank mode: `target`
- enabled module(s): `naive_joint=true`
- disabled module(s): `platform_token=false`, `shared_private=false`, `ot=false`, `gene_completion=false`, `cell_refine=false`

## Frozen formal metrics
- Visium overall Pearson: `0.3525680970`
- Visium mean gene Pearson: `0.0241456926`
- Xenium overall Pearson: `0.7532547695`
- Xenium mean gene Pearson: `0.3757945299`

## Rule for next step
The next stage/module experiment must add **exactly one** core change and be compared against this frozen Stage 2 baseline under matched formal settings.
