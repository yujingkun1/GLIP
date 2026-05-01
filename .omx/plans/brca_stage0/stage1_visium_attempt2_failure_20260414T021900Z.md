# Stage 1.1 失败记录：GLIP Visium-only baseline attempt 2

## 配置
- 模型：GLIP Visium-only
- 图像编码器：UNI2-h
- split：sample-level LOO
- batch_size：2
- 训练脚本：已支持 fold-level resume

## 新鲜证据
- 成功读取 36 个样本、13612 个 spots、785 个基因
- 成功识别并跳过已完成的 Fold 1（SPA119）
- 成功进入 Fold 2（SPA120 holdout）
- 在 epoch 1 优化器更新阶段再次触发 CUDA OOM

## 关键原因
不是数据/split 变化，而是当前 GPU0 上存在其他常驻进程，占用了较多显存；在这种环境下，batch_size=2 仍然不足够安全。

## 决策
- 该 attempt 不计入最终 baseline
- 下一步必须从头启动一个 **batch_size=1** 的全新 Stage 1 baseline run，确保全 36 folds 配置一致
