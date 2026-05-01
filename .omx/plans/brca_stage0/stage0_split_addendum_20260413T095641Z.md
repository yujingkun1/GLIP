# Stage 0 补充说明：基线复现 split 与严格泛化 split 分离

在继续 Stage 1 前，补充冻结如下原则：

## 1. 基线复现 split
为了与现有 GLIP / BLEEP / 文献中的 BRCA leave-one-out 结果保持可比性，
Stage 1 基线复现采用：

- **Visium：sample-level leave-one-out**
  - `visium_sample_loo_splits.json`
- **Xenium：NCBI784 x-position 5-fold，fold4 test**

这是“与已有 baseline 横向可比”的 split。

## 2. 严格泛化 split
为了支撑后续论文中更强的 generalization claim，额外保留：

- **Visium：patient-held-out**
  - `visium_patient_holdout_splits.json`
- **Xenium：后续如补到新的 BRCA donor，再做 donor-held-out；当前仅能 region-held-out**

## 3. 阶段执行规则
- Stage 1（baseline reproduction）：以 sample-level LOO 为主
- Stage 2 / Stage 3（核心模块增量比较）：
  - 首先沿用 Stage 1 的 sample-level LOO，保证和 baseline 完全可比
  - 核心结果站稳后，再追加 patient-held-out/generalization 评估

这样可同时满足：
1. 与已有 GLIP baseline 数字可比
2. 后续 generalization claim 更严格
