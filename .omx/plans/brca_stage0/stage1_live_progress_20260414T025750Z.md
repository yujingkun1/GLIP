# Stage 1 实时进展快照（Epoch 14）

## 数据集使用声明
当前仍然只使用：
- Visium：SPA119-SPA154
- Xenium：NCBI784
- 参考样本：SPA124（允许范围内）

未引入 NCBI785 或任何新下载数据。

## 新鲜证据
### Visium baseline
- 进程仍在运行
- fold_00_SPA119 目录目前仅有 `best.pt`
- `pearson_metrics.json` / `training_history.json` / `loo_fold_metrics.json` 尚未落盘
- 因此当前只能判断 baseline 仍在运行，不能记为 fold 完成

### Xenium pseudo-spot baseline
- metrics.json 已到 Epoch 14
- Epoch 13 test: overall 0.7671, gene 0.5130
- Epoch 14 test: overall 0.7642, gene 0.5220
- 历史最好：
  - overall 0.7704（Epoch 8）
  - gene 0.5591（Epoch 6）

## 当前结论
- Stage 1 继续阻塞于 Visium fold0 正式结果未落盘
- Xenium baseline 已稳定且结果充分
- 下一步优先继续等待 Visium 第一个正式 fold 结果；若长时间仍不落盘，则转入流程排查
