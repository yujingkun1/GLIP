# Unseen-data candidate freeze (interim) for GLIP BRCA Nature Communications experiments

## Purpose
Freeze the best currently supported unseen-data candidates before the full official HEST metadata audit completes, using fresh local evidence plus official HEST guidance.

## Official evidence
HEST official README (cached locally) explicitly supports selecting subsets by metadata filters such as:
- `organ == Breast`
- `oncotree_code == IDC`

Reference evidence file:
- `/data/yujk/GLIP/.omx/plans/hest_official_evidence_20260415T071504Z.md`

## Fresh local evidence
### Local candidate pool
- extra Visium candidates: 52
- extra Xenium candidates: 4

Source:
- `/data/yujk/GLIP/.omx/plans/local_hest_candidate_pool_20260415T071504Z.json`

### Local h5ad audit
Source:
- `/data/yujk/GLIP/.omx/plans/local_hest_candidate_h5ad_audit_20260415T071504Z.json`

Key observations:
- `NCBI785` Xenium has shape `4180 x 541`, closely matching core Xenium `NCBI784`
- `TENX120` and `TENX139` are also shape-compatible Xenium candidates
- `TENX157` has a much larger gene dimension (`4427 x 10006`) and is less suitable as the first external-test candidate
- Local extra Visium candidates include a coherent NCBI subgroup (`NCBI642`, `NCBI643`, `NCBI759`-`NCBI770`) that is a natural first-pass external-test pool pending official metadata confirmation

## Interim freeze decision
### Unseen Xenium external-test priority
1. **Primary candidate: `NCBI785`**
2. Secondary reserve candidates: `TENX120`, `TENX139`
3. Deferred candidate: `TENX157` (dimension mismatch risk)

### Unseen Visium external-test priority
1. **Primary review pool: `NCBI642`, `NCBI643`, `NCBI759`-`NCBI770`**
2. Secondary pool: `MEND139`-`MEND162`
3. Deferred pool: `TENX*` / `ZEN*` pending official disease/organ confirmation

## Rules until official metadata audit completes
- No new unseen candidate will be added to formal training yet
- `NCBI785` may be prepared as the first formal unseen Xenium external-test candidate
- Formal unseen Visium selection still requires official metadata confirmation of breast/IDC suitability


## User override (2026-04-15)
- Formal unseen-data generalization MUST use the same cancer type but a different tissue.
- User clarified local currently available candidates do NOT satisfy this requirement.
- Therefore the interim local-candidate freeze above is invalid for formal Experiment 2 execution.
- Next action: search official HEST data and download a same-cancer-type / different-tissue external-test dataset.
