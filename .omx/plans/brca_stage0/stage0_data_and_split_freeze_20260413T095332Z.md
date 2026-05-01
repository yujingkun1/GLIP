# Stage 0 记录：BRCA 数据与 split 冻结

## 已完成事项
1. 冻结核心数据范围：
   - Visium：SPA119-SPA154（HEST BRCA）
   - Xenium：NCBI784（核心）
   - NCBI785（仅作为候选外部 BRCA Xenium 测试）
2. 生成数据清单：
   - `visium_brca_dataset_manifest.csv`
   - `xenium_brca_dataset_manifest.csv`
3. 生成 split manifest：
   - `visium_patient_holdout_splits.json`
   - `xenium_core_split_manifest.json`
4. 生成 benchmark schema：
   - `benchmark_schema.json`
5. 生成患者-样本对照：
   - `patient_to_samples.json`

## 关键发现
### Visium BRCA
- 共 36 个样本
- 来自 8 个 patient（A-H）
- 患者级留出是合理的主 split 单位

### Xenium BRCA
- 本地 Breast Xenium 有 `NCBI784` 和 `NCBI785`
- 二者 metadata 显示 patient 都是 `patient 4`
- 因此 **NCBI785 不能作为严格 donor-level 外部泛化测试**
- 在当前核心范围内，Xenium 只能先做：
  - region-held-out / x-position-held-out
  - 后续若要写更强 donor-level Xenium 泛化，需补充新的 BRCA Xenium donor

## Stage 1 冻结协议
### Visium baseline
- 任务：GLIP Visium-only
- 数据：SPA119-SPA154
- split：patient-held-out（8 splits）
- 指标：overall Pearson, gene Pearson

### Xenium baseline
- 任务：GLIP Xenium-only / Xenium pseudo-spot baseline
- 数据：NCBI784
- split：x-position contiguous 5-fold，固定 fold4 为 test（先与 GLIP 默认对齐）
- 指标：overall Pearson, gene Pearson

## 晋级判定
Stage 0 -> Stage 1 已满足：
- 数据范围冻结
- split 规则冻结
- 指标字段冻结
- 结果总账 CSV 已建立

下一步进入 Stage 1：复现 GLIP baseline。
