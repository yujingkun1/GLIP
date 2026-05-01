# Stage 1 实时进展快照

## 当前无新增数据集
仍然只使用：
- Visium：SPA119-SPA154
- Xenium：NCBI784
- Xenium pseudo-spot 参考样本：SPA124（在允许范围内）

## 新鲜验证证据
### Visium baseline（clean run）
- run: `brca_stage1_glip_uni_sampleloo_bs2_clean`
- Fold 1 / 36 (`SPA119` holdout)
- 已持续推进超过 1300 / 6551 个训练 step
- 未再出现 OOM
- 训练 loss 已从约 8.81 下降到约 2.10

### Xenium pseudo-spot baseline
- run: `brca_stage1_ncbi784_pseudospot_spa124_bs8`
- Epoch 1 test: overall Pearson = 0.6833, gene Pearson = 0.3907
- Epoch 2 test: overall Pearson = 0.6973, gene Pearson = 0.3442
- 说明 baseline 已成功稳定跑通，不再是只起跑即 OOM

## 当前判断
- Stage 1 仍在进行，尚不能晋级
- 但关键阻塞（GPU 被旧任务占用 + batch OOM）已被部分解决
- 接下来继续等待：
  1. Visium clean run 完成首个 fold 的正式 pearson_metrics
  2. Xenium pseudo-spot 跑满更多 epoch并生成最终 metrics.json
