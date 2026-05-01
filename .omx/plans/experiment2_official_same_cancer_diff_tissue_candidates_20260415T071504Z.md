# Experiment 2 official candidates: same cancer type, different tissue

## Fresh evidence source
- Official local HEST metadata: `/data/yujk/hovernet2feature/HEST/assets/HEST_v1_1_0.csv`

## Core study cancer definition
- Core Visium set `SPA119-154`: `organ=Breast`, `oncotree_code=IDC`
- Core Xenium set `NCBI784`: `organ=Breast`, `oncotree_code=IDC`

Therefore the formal unseen-data generalization target is:
- **same cancer type**: `oncotree_code=IDC`
- **different tissue/organ**: not `Breast`

## Matching official HEST candidates found
The following samples satisfy the rule in the official metadata and are **not local yet**:

1. `NCBI681`
   - organ: `Lymph node`
   - oncotree_code: `IDC`
   - st_technology: `Visium`
   - dataset_title: `Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients`
   - subseries: `PT_8 LNM ST`

2. `NCBI682`
   - organ: `Lymph node`
   - oncotree_code: `IDC`
   - st_technology: `Visium`
   - dataset_title: `Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients`
   - subseries: `PT_7 LNM ST`

3. `NCBI683`
   - organ: `Lymph node`
   - oncotree_code: `IDC`
   - st_technology: `Visium`
   - dataset_title: `Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients`
   - subseries: `PT_6 LNM ST`

4. `NCBI684`
   - organ: `Lymph node`
   - oncotree_code: `IDC`
   - st_technology: `Visium`
   - dataset_title: `Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients`
   - subseries: `PT_3 LNM ST`

## Interim decision
- These four samples are the first formal download targets for Experiment 2.
- They satisfy the user hard gate better than any current local sample.
- Next step: download them from official HEST and freeze them as the formal unseen Visium generalization benchmark.

## User clarification (BRCA scope)
- The study cancer type is fixed to **BRCA / breast cancer**.
- Current core samples (`SPA119-154`, `NCBI784`) are all on the BRCA line.
- Therefore Experiment 2 should be interpreted as: **same cancer type = BRCA/IDC breast cancer**, **different tissue = non-breast tissue such as lymph node metastasis**.

## BRCA vs IDC wording note
- The study is on the **BRCA / breast cancer line**.
- In official HEST metadata, the relevant core samples are labeled with the more specific oncotree subtype **IDC** rather than the broader code `BRCA`.
- Therefore formal Experiment 2 should be described as: same BRCA line, specifically **IDC**, transferred from **Breast** tissue to **Lymph node** tissue.
