# Stage 1.1 Run Spec: GLIP Visium-only BRCA Sample-LOO Baseline

- repo: `/data/yujk/GLIP`
- task: reproduce GLIP Visium-only BRCA leave-one-out baseline
- samples: `SPA119`-`SPA154`
- split: sample-level leave-one-out
- gene file: `/data/yujk/hovernet2feature/HisToGene/data/her_hvg_cut_1000.txt`
- model: UNI2-h local checkpoint
- device: CUDA 0
- epochs: 10
- batch size: 8
- num workers: 4
- note: this run is the Stage 1 baseline reference for later one-module-at-a-time promotion tests
