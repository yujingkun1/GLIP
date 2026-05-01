# Experiment 2 hard gate

## User requirement (2026-04-15)
Formal unseen-data generalization must use:
1. the same cancer type as the core study
2. a different tissue

## Consequence
- Current local unseen-candidate freeze is invalid for the formal Experiment 2 benchmark.
- Official HEST search/download is required before formal Experiment 2 execution.
- Until then, unseen-data work remains in the data-audit phase.

## User clarification (BRCA scope)
- The study cancer type is fixed to **BRCA / breast cancer**.
- Current core samples (`SPA119-154`, `NCBI784`) are all on the BRCA line.
- Therefore Experiment 2 should be interpreted as: **same cancer type = BRCA/IDC breast cancer**, **different tissue = non-breast tissue such as lymph node metastasis**.

## BRCA vs IDC wording note
- The study is on the **BRCA / breast cancer line**.
- In official HEST metadata, the relevant core samples are labeled with the more specific oncotree subtype **IDC** rather than the broader code `BRCA`.
- Therefore formal Experiment 2 should be described as: same BRCA line, specifically **IDC**, transferred from **Breast** tissue to **Lymph node** tissue.
