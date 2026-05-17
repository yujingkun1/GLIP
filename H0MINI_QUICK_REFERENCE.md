# H0-mini 训练命令速查表

## 🚀 生产环境推荐命令（完整训练）

### 1. Visium Spot-level (30 epochs)

```bash
python train_visium.py \
    --model h0mini \
    --exp_name /data/yujk/GLIP/runs_visium/h0mini_30epochs \
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

### 2. Xenium Pseudo-spot (20 epochs)

```bash
python train_xenium_pseudospot.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_xenium/h0mini_20epochs \
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

### 3. Joint Training (30 epochs)

```bash
python train_joint_brca_naive.py \
    --model_name h0mini \
    --exp_name /data/yujk/GLIP/runs_joint/h0mini_30epochs \
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

---

## 📊 Epoch 数量建议

| 场景 | 推荐 Epochs | 预计训练时间 (单GPU) |
|------|------------|-------------------|
| Visium | 30-50 | 75-125 小时 |
| Xenium | 20-30 | 50-75 小时 |
| Joint | 30-50 | 75-125 小时 |

---

## 🔄 与 ResNet 的对比

只需将参数改为：
- `--model resnet50` → `--model h0mini` (Visium)
- `--model_name resnet50` → `--model_name h0mini` (Xenium/Joint)

其他参数完全相同！

---

## ⚡ 快速测试（1 epoch）

仅用于验证代码，不适合实际训练：

```bash
# Visium
python train_visium.py --model h0mini --max_epochs 1 --batch_size 32

# Xenium  
python train_xenium_pseudospot.py --model_name h0mini --epochs 1 --batch_size 32

# Joint
python train_joint_brca_naive.py --model_name h0mini --epochs 1 --batch_size 32
```

---

## 📝 关键参数

- `--model h0mini` 或 `--model_name h0mini`：使用 H0-mini
- `--batch_size 32`：推荐批次大小
- `--epochs 30`：推荐训练轮数
- `--image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin`：权重路径

---

详细文档：`H0MINI_PREDICTION_COMMANDS.md`
