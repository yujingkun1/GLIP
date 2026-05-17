# H0-mini 规范化实验命令

本文档提供规范化的实验命令，包含清晰的命名规范和完整的实验记录。

## 📁 推荐的目录结构

```
/data/yujk/GLIP/
├── experiments/
│   ├── h0mini_visium/
│   │   ├── fold0_30epochs_20260512/
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   ├── config.json
│   │   │   └── results.json
│   │   ├── fold1_30epochs_20260512/
│   │   └── ...
│   ├── h0mini_xenium/
│   │   ├── rep1_20epochs_20260512/
│   │   └── ...
│   └── h0mini_joint/
│       ├── fold0_30epochs_20260512/
│       └── ...
```

---

## 🎯 规范化命令（推荐使用）

### 1. Visium Spot-level

#### 单个 Fold 训练

```bash
# 设置实验参数
DATE=$(date +%Y%m%d)
FOLD_INDEX=0
EPOCHS=30
EXP_NAME="h0mini_visium_fold${FOLD_INDEX}_${EPOCHS}ep_${DATE}"
RUN_DIR="/data/yujk/GLIP/experiments/h0mini_visium/${EXP_NAME}"

# 创建实验目录
mkdir -p ${RUN_DIR}

# 训练命令
python train_visium.py \
    --model h0mini \
    --exp_name ${RUN_DIR} \
    --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --max_epochs ${EPOCHS} \
    --num_workers 8 \
    --cv_mode fixed_manifest \
    --fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
    --fold_index ${FOLD_INDEX} \
    --pretrained true \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0 \
    2>&1 | tee ${RUN_DIR}/training.log

# 保存实验信息
cat > ${RUN_DIR}/experiment_info.txt << EOF
Experiment: H0-mini Visium Training
Date: ${DATE}
Model: H0-mini (frozen)
Task: Visium spot-level prediction
Fold: ${FOLD_INDEX}
Epochs: ${EPOCHS}
Batch Size: 32
Data: BRCA Visium (36 samples)
Genes: 227 shared genes
Notes: Baseline H0-mini experiment, no VAE
EOF
```

#### 5-Fold 交叉验证（完整实验）

```bash
#!/bin/bash
# 文件名: run_h0mini_visium_5fold.sh

DATE=$(date +%Y%m%d)
EPOCHS=30
BASE_DIR="/data/yujk/GLIP/experiments/h0mini_visium"

for FOLD in 0 1 2 3 4; do
    EXP_NAME="fold${FOLD}_${EPOCHS}ep_${DATE}"
    RUN_DIR="${BASE_DIR}/${EXP_NAME}"
    
    mkdir -p ${RUN_DIR}
    
    echo "=========================================="
    echo "Training Fold ${FOLD}"
    echo "Output: ${RUN_DIR}"
    echo "=========================================="
    
    python train_visium.py \
        --model h0mini \
        --exp_name ${RUN_DIR} \
        --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
        --gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
        --batch_size 32 \
        --max_epochs ${EPOCHS} \
        --num_workers 8 \
        --cv_mode fixed_manifest \
        --fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
        --fold_index ${FOLD} \
        --pretrained true \
        --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
        --device_id 0 \
        2>&1 | tee ${RUN_DIR}/training.log
    
    # 保存实验信息
    cat > ${RUN_DIR}/experiment_info.txt << EOF
Experiment: H0-mini Visium 5-Fold CV
Date: ${DATE}
Model: H0-mini (frozen)
Fold: ${FOLD}/5
Epochs: ${EPOCHS}
Batch Size: 32
Status: Completed
EOF
    
    echo "Fold ${FOLD} completed!"
    echo ""
done

echo "All 5 folds completed!"
echo "Results saved in: ${BASE_DIR}"
```

---

### 2. Xenium Pseudo-spot

```bash
# 设置实验参数
DATE=$(date +%Y%m%d)
EPOCHS=20
XENIUM_ID="Xenium_FFPE_Human_Breast_Cancer_Rep1"
EXP_NAME="h0mini_xenium_rep1_${EPOCHS}ep_${DATE}"
RUN_DIR="/data/yujk/GLIP/experiments/h0mini_xenium/${EXP_NAME}"

# 创建实验目录
mkdir -p ${RUN_DIR}

# 训练命令
python train_xenium_pseudospot.py \
    --model_name h0mini \
    --exp_name ${RUN_DIR} \
    --xenium_sample_id ${XENIUM_ID} \
    --reference_visium_sample_id SPA119 \
    --pseudo_output_base_dir /data/yujk/GLIP/pseudospot_output \
    --shared_gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --epochs ${EPOCHS} \
    --test_fold 4 \
    --num_position_folds 5 \
    --max_train_spots 50000 \
    --max_test_spots 10000 \
    --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0 \
    2>&1 | tee ${RUN_DIR}/training.log

# 保存实验信息
cat > ${RUN_DIR}/experiment_info.txt << EOF
Experiment: H0-mini Xenium Pseudo-spot
Date: ${DATE}
Model: H0-mini (frozen)
Task: Xenium pseudo-spot prediction
Sample: ${XENIUM_ID}
Epochs: ${EPOCHS}
Batch Size: 32
Test Fold: 4/5
Max Train Spots: 50000
Max Test Spots: 10000
Notes: Baseline H0-mini experiment, no VAE
EOF
```

---

### 3. Joint Training (Visium + Xenium)

```bash
# 设置实验参数
DATE=$(date +%Y%m%d)
FOLD_INDEX=0
EPOCHS=30
EXP_NAME="h0mini_joint_fold${FOLD_INDEX}_${EPOCHS}ep_${DATE}"
RUN_DIR="/data/yujk/GLIP/experiments/h0mini_joint/${EXP_NAME}"

# 创建实验目录
mkdir -p ${RUN_DIR}

# 训练命令
python train_joint_brca_naive.py \
    --model_name h0mini \
    --run-dir ${RUN_DIR} \
    --visium_hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
    --visium_sample_ids SPA119,SPA120,SPA121,SPA122,SPA123,SPA124,SPA125,SPA126,SPA127,SPA128,SPA129,SPA130,SPA131,SPA132,SPA133,SPA134,SPA135,SPA136,SPA137,SPA138,SPA139,SPA140,SPA141,SPA142,SPA143,SPA144,SPA145,SPA146,SPA147,SPA148,SPA149,SPA150,SPA151,SPA152,SPA153,SPA154 \
    --visium_fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
    --visium_fold_index ${FOLD_INDEX} \
    --xenium_sample_id Xenium_FFPE_Human_Breast_Cancer_Rep1 \
    --xenium_reference_visium_sample_id SPA119 \
    --pseudo_output_base_dir /data/yujk/GLIP/pseudospot_output \
    --shared_gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
    --batch_size 32 \
    --epochs ${EPOCHS} \
    --xenium_test_fold 4 \
    --xenium_num_position_folds 5 \
    --max_xenium_train_spots 50000 \
    --max_xenium_test_spots 10000 \
    --checkpoint_path /data/yujk/H0-mini/pytorch_model.bin \
    --device_id 0 \
    2>&1 | tee ${RUN_DIR}/training.log

# 保存实验信息
cat > ${RUN_DIR}/experiment_info.txt << EOF
Experiment: H0-mini Joint Training
Date: ${DATE}
Model: H0-mini (frozen)
Task: Joint Visium + Xenium training
Visium Fold: ${FOLD_INDEX}/5
Xenium Test Fold: 4/5
Epochs: ${EPOCHS}
Batch Size: 32
Visium Samples: 36 BRCA samples
Xenium Sample: Xenium_FFPE_Human_Breast_Cancer_Rep1
Genes: 227 shared genes
Notes: Baseline H0-mini joint experiment, no VAE
EOF
```

---

## 📊 实验命名规范

### 格式
```
{model}_{task}_{fold/sample}_{epochs}ep_{date}
```

### 示例
- `h0mini_visium_fold0_30ep_20260512`
- `h0mini_xenium_rep1_20ep_20260512`
- `h0mini_joint_fold0_30ep_20260512`
- `resnet50_visium_fold0_30ep_20260512` (对比实验)

---

## 📝 实验记录模板

每个实验目录应包含：

```
experiment_dir/
├── experiment_info.txt      # 实验元信息
├── training.log             # 完整训练日志
├── config.json              # 模型配置
├── checkpoints/             # 模型检查点
│   ├── best_model.pt
│   └── last_model.pt
├── logs/                    # 训练指标
│   ├── train_loss.csv
│   └── val_metrics.csv
└── results/                 # 评估结果
    ├── predictions.npy
    └── metrics.json
```

---

## 🔍 实验对比脚本

创建对比脚本 `compare_experiments.sh`：

```bash
#!/bin/bash
# 对比不同模型的实验结果

echo "=========================================="
echo "Experiment Comparison"
echo "=========================================="
echo ""

# H0-mini
echo "H0-mini Results:"
for fold in 0 1 2 3 4; do
    result_file="/data/yujk/GLIP/experiments/h0mini_visium/fold${fold}_30ep_*/results/metrics.json"
    if [ -f ${result_file} ]; then
        echo "  Fold ${fold}: $(cat ${result_file} | grep 'test_pearson' | head -1)"
    fi
done
echo ""

# ResNet50 (对比)
echo "ResNet50 Results:"
for fold in 0 1 2 3 4; do
    result_file="/data/yujk/GLIP/experiments/resnet50_visium/fold${fold}_30ep_*/results/metrics.json"
    if [ -f ${result_file} ]; then
        echo "  Fold ${fold}: $(cat ${result_file} | grep 'test_pearson' | head -1)"
    fi
done
```

---

## 🚀 快速启动脚本

保存为 `run_h0mini_experiment.sh`：

```bash
#!/bin/bash

# 使用方法:
# ./run_h0mini_experiment.sh visium 0 30
# ./run_h0mini_experiment.sh xenium - 20
# ./run_h0mini_experiment.sh joint 0 30

TASK=$1
FOLD=$2
EPOCHS=$3
DATE=$(date +%Y%m%d)

case ${TASK} in
    visium)
        EXP_NAME="h0mini_visium_fold${FOLD}_${EPOCHS}ep_${DATE}"
        RUN_DIR="/data/yujk/GLIP/experiments/h0mini_visium/${EXP_NAME}"
        mkdir -p ${RUN_DIR}
        python train_visium.py \
            --model h0mini \
            --exp_name ${RUN_DIR} \
            --max_epochs ${EPOCHS} \
            --fold_index ${FOLD} \
            --batch_size 32 \
            --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
            2>&1 | tee ${RUN_DIR}/training.log
        ;;
    xenium)
        EXP_NAME="h0mini_xenium_rep1_${EPOCHS}ep_${DATE}"
        RUN_DIR="/data/yujk/GLIP/experiments/h0mini_xenium/${EXP_NAME}"
        mkdir -p ${RUN_DIR}
        python train_xenium_pseudospot.py \
            --model_name h0mini \
            --exp_name ${RUN_DIR} \
            --epochs ${EPOCHS} \
            --batch_size 32 \
            --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
            2>&1 | tee ${RUN_DIR}/training.log
        ;;
    joint)
        EXP_NAME="h0mini_joint_fold${FOLD}_${EPOCHS}ep_${DATE}"
        RUN_DIR="/data/yujk/GLIP/experiments/h0mini_joint/${EXP_NAME}"
        mkdir -p ${RUN_DIR}
        python train_joint_brca_naive.py \
            --model_name h0mini \
            --run-dir ${RUN_DIR} \
            --epochs ${EPOCHS} \
            --visium_fold_index ${FOLD} \
            --batch_size 32 \
            --checkpoint_path /data/yujk/H0-mini/pytorch_model.bin \
            2>&1 | tee ${RUN_DIR}/training.log
        ;;
    *)
        echo "Usage: $0 {visium|xenium|joint} fold epochs"
        exit 1
        ;;
esac

echo "Experiment completed: ${RUN_DIR}"
```

使用方法：
```bash
chmod +x run_h0mini_experiment.sh
./run_h0mini_experiment.sh visium 0 30
```

---

## 📈 实验追踪建议

1. **使用 Git 记录**：
   ```bash
   cd /data/yujk/GLIP
   git add experiments/
   git commit -m "Add H0-mini fold0 results"
   ```

2. **创建实验日志**：
   ```bash
   echo "$(date): Started H0-mini Visium fold0" >> experiments/experiment_log.txt
   ```

3. **定期备份**：
   ```bash
   tar -czf h0mini_experiments_backup_$(date +%Y%m%d).tar.gz experiments/h0mini_*
   ```

---

## 总结

使用这些规范化命令的好处：
- ✅ 清晰的命名规范
- ✅ 完整的实验记录
- ✅ 自动保存日志
- ✅ 易于对比和复现
- ✅ 便于团队协作
