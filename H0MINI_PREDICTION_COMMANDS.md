# H0-mini 预测命令（Spot 精度，不带 VAE）

本文档提供使用 H0-mini 进行预测的命令，包括 Visium、Xenium 和 Joint 三种场景。

## 前提条件

确保已经完成 H0-mini 集成（运行 `python test_h0mini_integration.py` 验证）。

---

## 1. Visium Spot-level 预测

使用 H0-mini 替代 ResNet 进行 Visium 数据的训练和预测。

### 训练命令

```bash
python train_visium.py \
    --model h0mini \
    --exp_name /data/yujk/GLIP/runs_visium/hest_bleep_loo_h0mini \
    --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --max_epochs 30 \
    --num_workers 8 \
    --cv_mode leave_one_out \
    --pretrained true \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

### 参数说明

- `--model h0mini`：使用 H0-mini 编码器（关键参数）
- `--exp_name`：实验输出目录
- `--hest_data_dir`：HEST 数据根目录
- `--gene_file`：共享基因列表文件
- `--batch_size 32`：批次大小（H0-mini 比 ResNet 更省内存，可以用更大的 batch）
- `--max_epochs 30`：训练轮数（**推荐 30-50 epochs**）
- `--cv_mode leave_one_out`：留一法交叉验证
- `--image_encoder_checkpoint`：H0-mini 权重路径

### 关于 Epoch 数量的建议

根据不同场景，推荐的 epoch 数量：

| 场景 | 推荐 Epochs | 说明 |
|------|------------|------|
| **Visium (单样本测试)** | 30-50 | 数据量较小，需要充分训练 |
| **Visium (5-fold CV)** | 30-50 | 每个 fold 独立训练 |
| **Xenium (伪 spot)** | 20-30 | 数据量较大，收敛较快 |
| **Joint (Visium+Xenium)** | 30-50 | 联合训练需要更多 epochs |
| **快速测试** | 1-5 | 仅用于验证代码 |

**经验法则**：
- 观察验证集损失，如果还在下降就继续训练
- 使用 early stopping（patience=5-10）避免过拟合
- H0-mini 参数冻结时收敛较快，可以适当减少 epochs

### 使用固定 fold 的训练

```bash
python train_visium.py \
    --model h0mini \
    --exp_name /data/yujk/GLIP/runs_visium/hest_bleep_fold0_h0mini \
    --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --max_epochs 30 \
    --cv_mode fixed_manifest \
    --fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
    --fold_index 0 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

---

## 2. Xenium Pseudo-spot 预测

使用 H0-mini 进行 Xenium 伪 spot 的训练和预测。

### 训练命令

```bash
python train_xenium_pseudospot.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_xenium/pseudospot_h0mini \
    --xenium_sample_id Xenium_FFPE_Human_Breast_Cancer_Rep1 \
    --reference_visium_sample_id SPA119 \
    --pseudo_output_base_dir /data/yujk/GLIP/pseudospot_output \
    --shared_gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --epochs 20 \
    --test_fold 4 \
    --num_position_folds 5 \
    --max_train_spots 50000 \
    --max_test_spots 10000 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

### 参数说明

- `--model_name h0mini`：使用 H0-mini 编码器
- `--xenium_sample_id`：Xenium 样本 ID
- `--reference_visium_sample_id`：参考 Visium 样本（用于伪 spot 生成）
- `--pseudo_output_base_dir`：伪 spot 输出目录
- `--test_fold 4`：测试 fold（0-4）
- `--num_position_folds 5`：空间位置 fold 数量
- `--max_train_spots`：最大训练 spot 数量
- `--max_test_spots`：最大测试 spot 数量

---

## 3. Joint (Visium + Xenium) 预测

联合训练 Visium 和 Xenium 数据，使用 H0-mini 编码器。

### 训练命令

```bash
python train_joint_brca_naive.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_joint/visium_xenium_h0mini \
    --visium_hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --visium_sample_ids SPA119,SPA120,SPA121,SPA122,SPA123,SPA124,SPA125,SPA126,SPA127,SPA128,SPA129,SPA130,SPA131,SPA132,SPA133,SPA134,SPA135,SPA136,SPA137,SPA138,SPA139,SPA140,SPA141,SPA142,SPA143,SPA144,SPA145,SPA146,SPA147,SPA148,SPA149,SPA150,SPA151,SPA152,SPA153,SPA154 \
    --visium_fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
    --visium_fold_index 0 \
    --xenium_sample_id Xenium_FFPE_Human_Breast_Cancer_Rep1 \
    --xenium_reference_visium_sample_id SPA119 \
    --pseudo_output_base_dir /data/yujk/GLIP/pseudospot_output \
    --shared_gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --epochs 30 \
    --xenium_test_fold 4 \
    --xenium_num_position_folds 5 \
    --max_xenium_train_spots 50000 \
    --max_xenium_test_spots 10000 \
    --checkpoint_path /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

### 参数说明

- `--model_name h0mini`：使用 H0-mini 编码器
- `--visium_hest_data_dir`：Visium HEST 数据目录
- `--visium_sample_ids`：Visium 样本 ID 列表（逗号分隔）
- `--visium_fold_manifest`：Visium fold 配置文件
- `--visium_fold_index`：使用的 fold 索引
- `--xenium_sample_id`：Xenium 样本 ID
- `--xenium_reference_visium_sample_id`：参考 Visium 样本
- `--pseudo_output_base_dir`：伪 spot 输出目录
- `--checkpoint_path`：H0-mini 权重路径

### 不使用 VAE 的配置

默认情况下，这些命令都不使用 VAE。如果脚本有 VAE 相关参数，确保设置为：

```bash
--use_vae_decoder false \
--vae_recon_weight 0.0 \
--vae_kl_weight 0.0
```

---

## 与 ResNet 的对比

### ResNet50 命令（对比参考）

**Visium**:
```bash
python train_visium.py --model resnet50 ...
```

**Xenium**:
```bash
python train_xenium_pseudospot.py --model_name resnet50 ...
```

**Joint**:
```bash
python train_joint_brca_naive.py --model_name resnet50 ...
```

### H0-mini vs ResNet

| 特性 | ResNet50 | H0-mini |
|------|----------|---------|
| 参数量 | 25M | 86M |
| 输出维度 | 2048 | 768 |
| 预训练数据 | ImageNet | 43M 病理图像 |
| Batch size | 16-32 | 32-64（更省内存） |
| 归一化 | ImageNet | H0-mini 专用 |

---

## 快速测试命令

### 快速验证（1 epoch，仅用于测试代码）

**注意**：这些命令仅用于快速验证代码是否正常运行，不适合实际训练！

**Visium**:
```bash
python train_visium.py \
    --model h0mini \
    --exp_name /data/yujk/GLIP/test_h0mini_visium \
    --batch_size 32 \
    --max_epochs 1 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin
```

**Xenium**:
```bash
python train_xenium_pseudospot.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/test_h0mini_xenium \
    --batch_size 32 \
    --epochs 1 \
    --max_train_spots 1000 \
    --max_test_spots 200 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin
```

**Joint**:
```bash
python train_joint_brca_naive.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/test_h0mini_joint \
    --batch_size 32 \
    --epochs 1 \
    --max_spots_per_sample 500 \
    --max_xenium_train_spots 1000 \
    --checkpoint_path /data/yujk/H0-mini/pytorch_model.bin
```

---

## 生产环境推荐命令

### 完整训练（推荐用于实际实验）

**Visium (30 epochs)**:
```bash
python train_visium.py \
    --model h0mini \
    --exp_name /data/yujk/GLIP/runs_visium/h0mini_30epochs \
    --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --batch_size 32 \
    --max_epochs 30 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

**Xenium (20 epochs)**:
```bash
python train_xenium_pseudospot.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_xenium/h0mini_20epochs \
    --batch_size 32 \
    --epochs 20 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

**Joint (30 epochs)**:
```bash
python train_joint_brca_naive.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_joint/h0mini_30epochs \
    --batch_size 32 \
    --epochs 30 \
    --checkpoint_path /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0
```

---

## 常见问题

### Q: 如何确认使用了 H0-mini？

查看训练日志，应该看到：
```
Loading H0-mini encoder...
Model: ImageEncoder_H0mini
Output dimension: 768
Parameters frozen: True
```

### Q: 内存不足怎么办？

减小 batch size：
```bash
--batch_size 16  # 或更小
```

### Q: 如何使用多 GPU？

添加分布式训练参数：
```bash
--distributed \
--world_size 2 \
--init_method tcp://127.0.0.1:3456
```

### Q: 训练时间对比

在相同硬件上（单 GPU，NVIDIA A100/V100）：

| 模型 | 时间/epoch | 30 epochs 总时间 | 内存占用 |
|------|-----------|----------------|---------|
| ResNet50 | ~2 小时 | ~60 小时 | ~8GB |
| H0-mini (frozen) | ~2.5 小时 | ~75 小时 | ~10GB |
| UNI2-h (frozen) | ~3 小时 | ~90 小时 | ~12GB |

**注意**：
- H0-mini 参数冻结时训练较快
- 实际时间取决于数据集大小和硬件配置
- 使用更大的 batch size 可以加速训练

---

## 输出文件

训练完成后，输出目录包含：

```
exp_name/
├── checkpoints/
│   ├── best_model.pt          # 最佳模型
│   └── last_model.pt          # 最后一个 epoch
├── logs/
│   ├── train_log.json         # 训练日志
│   └── eval_results.json      # 评估结果
└── predictions/
    ├── train_embeddings.npy   # 训练集 embeddings
    └── test_predictions.npy   # 测试集预测
```

---

## 下一步

训练完成后，可以：

1. **评估模型性能**：查看 `eval_results.json`
2. **提取特征**：使用训练好的模型提取图像特征
3. **可视化**：使用 t-SNE/UMAP 可视化 embeddings
4. **对比实验**：与 ResNet50/UNI2-h 对比性能

---

## 总结

只需将原来的 `--model resnet50` 或 `--model_name resnet50` 替换为 `h0mini`，其他参数保持不变即可使用 H0-mini！
