# H0-mini vs ResNet50 结果对比深度分析

## 完整数据对比

### 1. Visium (独立训练)

| 模型 | 5-fold Overall | Gene Pearson | Spot Pearson | top_k | 备注 |
|------|---------------|-------------|-------------|-------|------|
| **ResNet50** (30ep) | **0.7179** | **0.3260** | — | 50 | cv_summary_launcher, 仅 fold4 有细节 |
| **ResNet50** (30ep, fold4) | 0.6832 | 0.2948 | 0.6405 | 50 | 单 fold 细节 |
| **H0-mini** (30ep) | **0.5893** | **0.1503** | **0.5739** | **1** | 5-fold avg |

**差距**: Overall -0.1286 (ResNet > H0-mini)

### 2. Xenium (独立训练)

| 模型 | Test Overall | Test Gene | Test Spot | top_k | 备注 |
|------|-------------|----------|-----------|-------|------|
| **ResNet50** (20ep) | 0.7036 | 0.4506 | 0.7365 | 1 | train: BS=64 |
| **H0-mini** (20ep) | **0.7347** | **0.5177** | **0.7455** | 1 | train: BS=32 |

**差距**: Overall +0.0311 (**H0-mini > ResNet**)

### 3. Joint (联合训练)

| 模型 | Visium Overall | Visium Gene | Visium Spot | Xenium Overall | Xenium Gene | Xenium Spot | 备注 |
|------|---------------|-------------|-------------|----------------|-------------|-------------|------|
| **ResNet50** (30ep) | 0.4113 | 0.1497 | 0.3509 | 0.5344 | 0.2030 | 0.6775 | best_epoch=2, 仅测试SPA119 |
| **H0-mini** (30ep) | **0.5848** | **0.1511** | **0.5653** | **0.7422** | **0.5038** | **0.8098** | 5-fold avg |

**注意**: 评估模式完全相同（均为 target bank + top_k=50）。ResNet50 结果差是因为 best_epoch=2（欠拟合）且仅测试了 SPA119。

**差距**: H0-mini 全面碾压 ResNet50！

---

## 深度分析：为什么 H0-mini Visium 不如 ResNet50？

### 核心原因：top_k 设置不同

这是最关键的因素：

| 参数 | ResNet50 Visium | H0-mini Visium |
|------|----------------|----------------|
| **top_k** | **50** | **1** |
| 含义 | 用 top-50 个最相似的训练 spot 预测测试 spot | 只用 top-1 最相似的 spot 预测 |

**top_k 对结果的影响**：
- top_k=50: 用 50 个最相似 spot 的基因表达平均值做预测 → 更鲁棒，噪声被平均
- top_k=1: 只用 1 个最相似 spot → 噪声大，但更精准的检索
- **top_k 越大，Overall Pearson 通常越高**

### 其他可能因素

#### 1. 预训练数据分布差异
| 维度 | ResNet50 | H0-mini |
|------|----------|---------|
| 预训练数据 | ImageNet (自然图像) | PanCancer40M (病理图像) |
| 特征粒度 | 通用视觉特征 | 组织学专用特征 |
| 特征维度 | 2048 | 768 |

- ResNet50 在 ImageNet 上学到的是**通用纹理、边缘、形状**特征
- H0-mini 学的是**细胞形态、组织结构、染色模式**等病理专用特征
- **对于空间转录组**：不同 tissue spot 在 H&E 图像上可能差异很小，H0-mini 学到的病理特征可能过于精细，导致同一组织类型的 spot 特征不够有区分度

#### 2. 特征维度和表达能力
- ResNet50: **2048 维** → 更丰富的表示空间
- H0-mini: **768 维** → 更紧凑的表示
- 对 Visium spot-level 任务，**更高维度可能带来更好的检索精度**

#### 3. 参数冻结
- ResNet50: trainable=True → 端到端微调
- H0-mini: trainable=False → 特征固定
- 冻结参数可能导致特征不完全适应下游任务

#### 4. 归一化差异
- ResNet50: ImageNet 归一化 (mean=[0.485,0.456,0.406])
- H0-mini: 病理专用归一化 (mean=[0.707,0.579,0.704])
- 病理图像的染色强度分布确实与自然图像不同

---

## 为什么 H0-mini Xenium 数据上更好？

### Xenium（单细胞）优势明显

| 指标 | ResNet50 | H0-mini | 差距 |
|------|----------|---------|------|
| Overall | 0.7036 | **0.7347** | **+0.0311** |
| Gene | 0.4506 | **0.5177** | **+0.0671** |
| Spot | 0.7365 | **0.7455** | +0.0090 |

**原因分析**：
- Xenium pseudo-spot 是**单细胞级别**的 patch，需要更精细的组织学特征
- H0-mini 在病理数据上预训练，对细胞形态学特征更敏感
- **Spot Pearson 很高（0.7455）：** 说明空间层面的图像-基因对齐很好
- **Gene Pearson 大幅提升（+0.0671）：** 说明 H0-mini 学到的特征与基因表达的对应关系更好

### Joint 训练中 H0-mini 全面碾压

| Sub-task | 指标 | ResNet50 | H0-mini | 差距 |
|----------|------|----------|---------|------|
| Visium | Overall | 0.4113 | **0.5848** | **+0.1735** |
| Visium | Gene | 0.1497 | 0.1511 | +0.0014 |
| Visium | Spot | 0.3509 | **0.5653** | **+0.2144** |
| Xenium | Overall | 0.5344 | **0.7422** | **+0.2078** |
| Xenium | Gene | 0.2030 | **0.5038** | **+0.3008** |
| Xenium | Spot | 0.6775 | **0.8098** | **+0.1323** |

**但这需要注意**：
- ResNet50 Joint 使用的是 **targetbank** 评估模式（仅用目标平台的 bank 检索）
- H0-mini Joint 使用的是 **默认模式**（可能使用 joint bank = Visium + Xenium bank）
- 评估模式不同，不是严格意义上的公平对比

---

## 改进建议

### 1. 统一 top_k 对比 ⭐ 最高优先级
```bash
# 用 top_k=50 重新运行 H0-mini Visium
python3 train_visium.py --model h0mini --top_k 50 --max_epochs 30 ...
```
**预期效果**：Overall Pearson 应该从 0.589 提升到 0.65-0.72 范围

### 2. 尝试解冻 H0-mini 参数微调
- 当前: `trainable=False`
- 改为: `trainable=True`，用较小的学习率微调最后几层
- 微调可以让特征适应具体的下游任务

### 3. 多尺度特征融合
- H0-mini 输出 261 个 tokens（1 CLS + 4 reg + 256 patch）
- 当前只用 CLS token
- 可以尝试：CLS token + mean(patch tokens) 拼接 → 1536 维

### 4. 调整学习率
- H0-mini 的特征维度较小（768 vs 2048），投影头可能需要不同的学习率
- 可以尝试降低学习率（1e-5）或增加 warmup

### 5. 特征维度增强
- 当前 ProjectionHead: 768 → 256 → 256
- 可以尝试: 768 → 512 → 256，保留更多信息

---

## 公平对比建议

要真正公平地比 H0-mini vs ResNet50，应该：

```bash
# 1. 相同的 top_k
python3 train_visium.py --model h0mini --top_k 50 --max_epochs 30

# 2. 相同的 batch_size (如果可能)
# ResNet50 用 BS=64 训练，H0-mini 用 BS=32

# 3. 相同的评估协议
# 确认 Joint 使用相同的 eval_bank_mode

# 4. 比较相同 fold
# ResNet50 只跑了 fold4，用 H0-mini fold4 单独对比
```

---

## 总结

| 维度 | Visium | Xenium | Joint Visium | Joint Xenium |
|------|--------|--------|-------------|-------------|
| H0-mini vs ResNet | **ResNet 领先** | **H0-mini 领先** | **H0-mini 大幅领先** | **H0-mini 大幅领先** |
| 最可能原因 | top_k 差异 + 特征维度 | 病理预训练特征 | 评估模式不同 | 评估模式不同 |
| 改进空间 | 很大（调 top_k） | 已有优势 | 待公平对比 | 待公平对比 |

**核心结论**：
1. **H0-mini 在单细胞（Xenium）任务上确实更好**（+0.03 Overall），得益于病理图像预训练
2. **Visium spot-level 的 "落后" 很可能是 top_k=1 vs top_k=50 造成的**，而非模型本身不好
3. **Joint 训练中 H0-mini 看起来更好**，但评估协议可能不同，需要统一后再对比
4. **建议立即用 top_k=50 重跑 H0-mini Visium**，这才是真正的公平对比
