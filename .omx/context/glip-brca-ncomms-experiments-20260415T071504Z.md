# GLIP BRCA Nature Communications experiments context snapshot

## task statement
Find official HEST unseen-data candidates, update the experiment plan, then start the GLIP BRCA Visium/Xenium experiment program step by step under Ralph.

## desired outcome
A grounded execution lane with approved experiment structure, official unseen-data candidates, and PRD/test-spec artifacts ready before implementation/experiments proceed.

## known facts/evidence
- Core training scope frozen to Visium SPA119-154 and Xenium NCBI784.
- Local HEST Visium files include SPA119-154 plus additional NCBI/TENX/ZEN/MEND samples.
- Local HEST Xenium files include NCBI784, NCBI785, TENX120, TENX139, TENX157.
- Plan v3 exists at /data/yujk/GLIP/.omx/plans/plan-v3-ncomms-experiments-20260415T071504Z.md.
- User requires Nature Communications quality, full-run-only comparisons, and 5 core experiment families.

## constraints
- Do not use smoke results as formal comparison evidence.
- If new datasets are introduced, disclose source and role clearly.
- Prefer HEST official/internal data for unseen-batch generalization.
- Stage/module progression should remain incremental and controlled.

## unknowns/open questions
- Which HEST samples are the best unseen breast/generalization candidates.
- Whether local HEST already contains sufficient external breast candidates or if download is required.
- Exact threshold for Stage2 catastrophic regression.

## likely codebase touchpoints
- /data/yujk/GLIP/.omx/plans/
- /data/yujk/GLIP/train_visium.py
- /data/yujk/GLIP/train_joint_brca_naive.py
- /data/yujk/GLIP/processed/
- /data/yujk/hovernet2feature/HEST/
