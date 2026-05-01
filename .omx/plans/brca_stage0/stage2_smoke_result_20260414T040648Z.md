# Stage 2 smoke 结果：naive joint Visium + Xenium pseudo-spot

## 数据集使用声明
当前仍然只使用：
- Visium：SPA119-SPA154
- Xenium：NCBI784
- 参考样本：SPA124（允许范围内）

未引入 NCBI785 或任何外部下载数据。

## 本阶段唯一新增核心
- 只新增一个核心：
  - 在共享 313-gene space 中，将 Visium spot 与 Xenium pseudo-spot 直接混合训练一个共享 retrieval 模型
- 未加入：
  - shared/private latent
  - OT
  - completion
  - cell-level refinement

## smoke 配置
- heldout Visium：SPA119
- Visium train/test capped：256 / 128
- Xenium train/test capped：128 / 64
- epochs：1
- top_k：10

## smoke 结果
- Visium overall Pearson：0.4087
- Visium gene Pearson：0.0068
- Xenium overall Pearson：0.6900
- Xenium gene Pearson：0.4039

## 结论
- Stage 2 最小 pipeline 已成功打通
- 但该 smoke 结果显著不足以超越已接受 baseline
- 因此 Stage 2 **不能晋级**
- 下一步应继续在 Stage 2 内调参/扩大训练规模（仍不引入新模块），而不是进入 Stage 3
