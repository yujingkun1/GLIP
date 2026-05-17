# H0-mini 训练分析与调参建议

## 1. Visium unfrozen：明显过拟合

### 训练曲线 (fold0)

| Epoch | Train Loss | 变化 |
|-------|-----------|------|
| 1 | 6.53 | - |
| 5 | 2.33 | ↓ |
| 10 | 1.66 | ↓ |
| 15 | 1.19 | ↓ |
| 20 | 0.73 | ↓↓ |
| 25 | 0.43 | ↓↓ |
| 30 | **0.29** | ⚠️ 极低! |

### 当前结果 vs ResNet50

| 指标 | ResNet50 | H0-mini frozen | H0-mini unfrozen |
|------|----------|---------------|-----------------|
| **Gene Pearson** | 0.3260 | 0.1343 | **0.2993** ✅ |
| Overall | 0.7179 | 0.5444 | 0.6928 |
| Spot | — | 0.5425 | 0.6387 |

**结论**：解冻后 Gene Pearson 大幅回升（0.1343→0.2993），接近 ResNet50（0.3260），但 train loss 降到 0.29 明显过拟合。

### 过拟合原因 & 解决方案

**原因**：
- 30 epochs 太长，H0-mini 参数多（86M），在 Visium 数据量（~13K spots）上容易过拟合
- ResNet50 30epo 的 best_epoch=8，说明 ResNet 也过拟合，但保留了 best checkpoint
- H0-mini 当前脚本没有 early stopping，没有 per-epoch test eval

**解决方案**（按优先级）：

#### 方案 A：降低 epochs + 增加 weight_decay ⭐ 推荐
```bash
# weight_decay 从 1e-3 提高到 1e-2，epochs 降到 15-20
./run_h0mini_experiment.sh visium 15 trainable
# 手动加上 --weight_decay 1e-2
```
- `weight_decay=1e-2`：更强的正则化
- `epochs=15`：在过拟合前停住

#### 方案 B：降低学习率
```bash
# lr 从 1e-4 降到 1e-5
```
- H0-mini 预训练权重质量高，微调不需要大学习率

#### 方案 C：分层解冻（只解冻最后几层）
- 当前：全解冻
- 改进：只解冻最后 6 层 transformer blocks

#### 方案 D：添加 early stopping（需改代码）
- patience=5，监控 test loss，自动保存最佳模型

---

## 2. Xenium unfrozen：正常，但任务是伪 spot，不是细胞级别

### 训练曲线 (20 epochs)

| Epoch | Test Gene | Test Overall |
|-------|----------|-------------|
| 1 | 0.5098 | 0.7394 |
| 5 | 0.4549 | 0.7480 |
| 10 | 0.6252 | 0.8128 |
| 15 | 0.6172 | 0.8164 |
| 20 | **0.6283** | **0.8223** |

**结论**：没有明显过拟合。Gene Pearson 从 0.51→0.63，波动正常。Xenium pseudo-spot 只有 284 train / 76 test spots，数据量小而稳定。

### 当前结果对比

| 指标 | ResNet50 | H0-mini frozen | H0-mini unfrozen |
|------|----------|---------------|-----------------|
| Gene | 0.4506 | 0.6844 | **0.6283** |
| Overall | 0.7036 | 0.8426 | 0.8223 |

⚠️ 注意：H0-mini 用了 top_k=50，ResNet50 用 top_k=1，不能直接对比！

---

## 3. Xenium 5-fold 细胞级别：完全不同的任务！

### 你需要的是 `train_xenium.py`，不是 `train_xenium_pseudospot.py`

| 维度 | pseudo-spot (当前) | cell-level (你需要的) |
|------|-------------------|---------------------|
| 脚本 | `train_xenium_pseudospot.py` | `train_xenium.py` |
| 模块 | `glip.xenium.train_pseudospot` | `glip.xenium.train` |
| 训练单元 | 284 pseudo-spots | **77868 train cells** |
| 测试单元 | 76 pseudo-spots | **40842 test cells** |
| top_k | 50 (我们设的) | **1** |
| ResNet50 Gene | — | 0.2377 (fold00) |
| ResNet50 5-fold Overall | — | 0.5306 |

### ResNet50 细胞级别 5-fold 参考结果

```
brca_stage4_cell_xenium_only_resnet50_g227_5fold/
├── fold_00/  test_fold=0, Test Gene=0.2377, Overall=0.5497
├── fold_01/  test_fold=1
├── fold_02/  test_fold=2
├── fold_03/  test_fold=3
└── fold_04/  test_fold=4
5-fold avg Overall: 0.5306, avg Gene: 0.2383
```

**要做 Cell-level 5-fold，需要：**
1. 给 `glip/xenium/train.py` 添加 H0-mini 支持
2. 用 `train_xenium.py` 跑 5-fold（test_fold=0..4）

---

## 4. 调参建议汇总

### Visium（立即行动）

```bash
# 推荐：降低 epochs + 加强正则化
python train_visium.py \
    --model h0mini \
    --max_epochs 15 \
    --batch_size 32 \
    --top_k 50 \
    --trainable \
    --weight_decay 1e-2 \
    --lr 5e-5 \
    ...
```

### Xenium cell-level 5-fold（需要先加 H0-mini 支持）

这需要修改 `glip/xenium/train.py` 和 `glip/xenium/model.py`（cell-level ContrastiveImageGeneModel）。

### Xenium pseudo-spot（当前结果已可用）

当前 unfrozen 结果 Gene=0.6283 已经很好。如果要和 ResNet50 公平对比，改 top_k=1 即可。

---

## 5. 优先级

| 优先级 | 事项 | 说明 |
|--------|------|------|
| ⭐1 | Visium：epochs=15, wd=1e-2, lr=5e-5 | 解决过拟合 |
| ⭐2 | Xenium cell-level：给 `train_xenium.py` 加 H0-mini | 真正的 5-fold |
| ⭐3 | Xenium pseudo-spot：top_k=1 公平对比 | 参数对齐 |
