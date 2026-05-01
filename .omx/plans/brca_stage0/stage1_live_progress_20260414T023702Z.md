# Stage 1 实时进展快照（继续）

## 数据集使用声明
当前仍然只使用：
- Visium：SPA119-SPA154
- Xenium：NCBI784
- Xenium pseudo-spot 参考样本：SPA124（在允许范围内）

未引入 NCBI785 或任何新下载的外部数据。

## 新鲜证据
### Visium baseline（clean run）
- Fold 1 / 36 (`SPA119` holdout)
- 当前训练已推进至约 3998 / 6551 step
- train loss 下降到约 1.22
- 仍无新的 OOM
- 说明当前 clean baseline 可持续运行

### Xenium pseudo-spot baseline
- Epoch 5 test overall Pearson = 0.7202
- Epoch 5 test gene Pearson = 0.4545
- 相比 Epoch 1 的 0.6833 / 0.3907，整体仍保持提升

## 当前结论
- Stage 1 还未完成，不能晋级
- 但两个 baseline 都已稳定进入“持续训练并产生有效指标”的状态
