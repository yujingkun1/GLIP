# Stage 1.1 失败记录：GLIP Visium-only baseline attempt 1

## 运行配置
- 模型：GLIP Visium-only
- 图像编码器：UNI2-h
- 数据：SPA119-SPA154
- split：sample-level LOO
- batch_size：8
- epochs：10
- GPU：0

## 新鲜证据
本次实际启动后，数据集成功构建：
- 36 个 BRCA Visium 样本
- 13612 个对齐 spots
- 785 个基因
- 成功进入 Fold 1 / 36（SPA119 holdout）

但在 Fold 1 epoch 1 的首轮训练即报错：
- `torch.OutOfMemoryError`
- 4090 24GB 显存不足以支撑当前配置

## 结论
- Stage 1 尚未完成，不能晋级
- 下一步必须在 **相同数据、相同 split、相同模型设定** 下只调整资源参数：
  - 优先减小 batch size
  - 必要时降低 eval/query 规模或改用更保守的显存策略
- 不允许跳到下一阶段
