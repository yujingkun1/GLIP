# Test Spec: GLIP BRCA Visium/Xenium Nature Communications experiment program

## Formal evidence requirements
- All paper-grade comparisons must come from full runs
- Every formal run must record dataset scope, split, config, run_dir, and commit_sha
- No smoke run may be used as formal comparative evidence

## Experiment-family acceptance checks
### 1. Joint vs single-modality
- Report Visium and Xenium held-out metrics for single-modality and joint models
- Confirm the comparison uses matched formal data/split settings

### 2. Unseen-data generalization
- Candidate dataset provenance must be documented
- Verify no same-batch/same-patient leakage for formal external tests
- Report the same formal metrics on unseen data

### 3. Gene completion
- Define mask ratio(s), mask policy, and evaluation metrics before execution
- Compare completion quality between single-modality and joint models

### 4. Cell-level decoder
- Report cell-level metrics on Xenium
- Also verify whether primary spot/pseudospot tasks regress materially

### 5. Biological significance
- Use quantitative analysis, not only qualitative figures
- Compare predicted vs observed expression-derived biological signals

## Blocking items to resolve before full execution breadth
- Choose official unseen-data candidates
- Define catastrophic regression threshold
- Freeze gene-mask design
- Freeze cell-level decoder evaluation protocol
- Freeze biological analysis protocol
