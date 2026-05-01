# Stage 1.2 失败记录：Xenium pseudo-spot baseline attempt 1

## 配置
- 数据：NCBI784
- 参考 Visium 样本：SPA124（仍在允许的 SPA119-154 范围内）
- split：x-position 5-fold，fold4 test
- 模型：GLIP Xenium pseudo-spot + UNI2-h
- batch_size：32

## 新鲜证据
- 成功加载 pseudo-spot cache：`NCBI784_ref_SPA124`
- Train pseudo-spots: 284
- Test pseudo-spots: 76
- Gene dim: 313
- 在 Epoch 1 首轮训练时发生 CUDA OOM

## 决策
- 保持相同数据与 split
- 仅降低 batch size 继续 Stage 1
