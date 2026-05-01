# Stage 1 补丁记录：Visium baseline 支持断点续跑

## 修改内容
对 `/data/yujk/GLIP/train_visium.py` 做了最小基础设施修改：
- 若 `loo_fold_metrics.json` 已存在，则读取已完成 fold 的 `heldout_sample`
- 重跑时自动跳过已完成 fold
- 不改变模型结构、损失函数、数据切分或评价方式

## 目的
Stage 1 sample-level LOO 在 36 个样本上运行时间较长，且中途中断会浪费已完成 fold。该补丁仅用于保证 baseline 复现可持续推进。

## 影响范围
- 仅影响训练流程控制
- 不影响实验可比性
- 不属于方法模块变化，不计入晋级模块
