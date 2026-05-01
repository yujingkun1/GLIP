# PRD: GLIP BRCA Visium/Xenium Nature Communications experiment program

## Objective
Deliver a Nature Communications-grade BRCA cross-platform spatial transcriptomics study centered on GLIP-style modeling, using rigorous full-run comparisons and five core experiment families: joint-vs-single-modality performance, unseen-data generalization, gene completion, cell-level decoder evaluation, and biological significance validation.

## Core scope
- Core training Visium: SPA119-154
- Core training Xenium: NCBI784
- Unseen/generalization data: prioritize official HEST breast candidates outside the core set

## Must-have experiment families
1. Joint Visium+Xenium vs single-modality training on held-out Visium and Xenium tests
2. Unseen-batch/unseen-dataset generalization on official HEST candidates
3. Random-mask gene completion / cross-platform completion
4. Xenium cell-level decoder head evaluation
5. Biological significance analyses: pathway, biomarker, survival/clinical trend if data supports it

## Quality bar
- Formal comparisons require full runs only
- Exploratory runs can guide decisions but cannot support main claims
- Every promoted module must be justified against the experiment questions, not just technical novelty

## Immediate execution sequence
1. Audit official HEST candidates and local availability
2. Freeze unseen-data generalization candidate list
3. Freeze test spec and experiment pass criteria
4. Begin module-by-module experiments starting from the lowest-complexity core lane
