#!/bin/bash
# H0-mini 实验快速启动脚本
# 使用方法:
#   ./run_h0mini_experiment.sh visium 30      # 自动运行 5-fold
#   ./run_h0mini_experiment.sh xenium 20      # Xenium
#   ./run_h0mini_experiment.sh joint 30       # Joint 5-fold

set -e  # 遇到错误立即退出

# 设置 Python 路径（使用 conda 环境）
PYTHON="/home/yujk/miniconda3/bin/python"

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 {visium|xenium|joint} epochs [trainable]"
    echo ""
    echo "Examples:"
    echo "  $0 visium 30           # Visium 5-fold CV, 30 epochs, frozen (top_k=50)"
    echo "  $0 visium 30 trainable # Visium 5-fold CV, 30 epochs, unfrozen (top_k=50)"
    echo "  $0 xenium 20           # Xenium, 20 epochs, frozen"
    echo "  $0 xenium 20 trainable # Xenium, 20 epochs, unfrozen"
    echo "  $0 joint 30            # Joint 5-fold CV, 30 epochs, frozen"
    echo "  $0 joint 30 trainable  # Joint 5-fold CV, 30 epochs, unfrozen"
    exit 1
fi

TASK=$1
EPOCHS=$2
TRAINABLE=${3:-false}
if [ "$TRAINABLE" = "trainable" ]; then
    TRAINABLE_FLAG="--trainable"
    TRAINABLE_LABEL="unfrozen"
else
    TRAINABLE_FLAG=""
    TRAINABLE_LABEL="frozen"
fi
DATE=$(date +%Y%m%d_%H%M%S)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "H0-mini Experiment Launcher"
echo -e "==========================================${NC}"
echo "Task: ${TASK}"
echo "Epochs: ${EPOCHS}"
echo "Date: ${DATE}"
echo ""

case ${TASK} in
    visium)
        echo -e "${BLUE}Running Visium 5-Fold Cross-Validation (top_k=50, ${TRAINABLE_LABEL})${NC}"
        echo ""

        # 创建实验大文件夹（所有 folds 共享）
        EXP_BASE_NAME="h0mini_visium_5fold_${EPOCHS}ep_${TRAINABLE_LABEL}_${DATE}"
        EXP_BASE_DIR="/data/yujk/GLIP/experiments/h0mini_visium/${EXP_BASE_NAME}"
        mkdir -p ${EXP_BASE_DIR}

        # 保存实验总体信息
        cat > ${EXP_BASE_DIR}/experiment_summary.txt << EOF
Experiment: H0-mini Visium 5-Fold Cross-Validation
Date: ${DATE}
Model: H0-mini (${TRAINABLE_LABEL})
Task: Visium spot-level prediction
Total Folds: 5
Epochs per Fold: ${EPOCHS}
Batch Size: 32
top_k: 50
Data: BRCA Visium (36 samples)
Genes: 227 shared genes
Notes: Baseline H0-mini experiment, no VAE, top_k=50, ${TRAINABLE_LABEL}
Command: $0 $@
EOF

        for FOLD in 0 1 2 3 4; do
            FOLD_DIR="${EXP_BASE_DIR}/fold${FOLD}"

            echo -e "${YELLOW}=========================================="
            echo "Training Fold ${FOLD}/5"
            echo -e "==========================================${NC}"
            echo "Output: ${FOLD_DIR}"
            echo ""

            mkdir -p ${FOLD_DIR}

            ${PYTHON} train_visium.py \
                --model h0mini \
                --exp_name ${FOLD_DIR} \
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
                --top_k 50 \
                --device_id 0 \
                ${TRAINABLE_FLAG} \
                2>&1 | tee ${FOLD_DIR}/training.log

            # 保存每个 fold 的信息
            cat > ${FOLD_DIR}/fold_info.txt << EOF
Fold: ${FOLD}/5
Epochs: ${EPOCHS}
Status: Completed at $(date)
EOF

            echo -e "${GREEN}Fold ${FOLD} completed!${NC}"
            echo ""
        done

        echo -e "${GREEN}=========================================="
        echo "All 5 folds completed!"
        echo -e "==========================================${NC}"
        echo "Results saved to: ${EXP_BASE_DIR}"

        # 计算 5-fold 汇总统计
        echo ""
        echo -e "${YELLOW}Computing 5-fold summary statistics...${NC}"
        ${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${EXP_BASE_DIR}
        ;;

    xenium)
        EXP_NAME="h0mini_xenium_ncbi784_${EPOCHS}ep_${TRAINABLE_LABEL}_${DATE}"
        RUN_DIR="/data/yujk/GLIP/experiments/h0mini_xenium/${EXP_NAME}"

        echo -e "${YELLOW}Creating experiment directory...${NC}"
        mkdir -p ${RUN_DIR}

        echo -e "${YELLOW}Starting Xenium training (${TRAINABLE_LABEL})...${NC}"
        ${PYTHON} train_xenium_pseudospot.py \
            --model h0mini \
            --run-dir ${RUN_DIR} \
            --sample-id NCBI784 \
            --reference-visium-sample-id SPA124 \
            --pseudo-output-base-dir /data/yujk/GLIP/processed/pseudospots \
            --batch-size 32 \
            --epochs ${EPOCHS} \
            --test-fold 4 \
            --num-position-folds 5 \
            --max-train-spots 50000 \
            --max-test-spots 10000 \
            --image-encoder-checkpoint /data/yujk/H0-mini/pytorch_model.bin \
            --top-k 50 \
            --device cuda:1 \
            ${TRAINABLE_FLAG} \
            2>&1 | tee ${RUN_DIR}/training.log

        # 保存实验信息
        cat > ${RUN_DIR}/experiment_info.txt << EOF
Experiment: H0-mini Xenium Pseudo-spot
Date: ${DATE}
Model: H0-mini (${TRAINABLE_LABEL})
Task: Xenium pseudo-spot prediction
Sample: NCBI784
Reference: SPA124
Epochs: ${EPOCHS}
Batch Size: 32
Test Fold: 4/5
Max Train Spots: 50000
Max Test Spots: 10000
Notes: Baseline H0-mini experiment, no VAE, top_k=50, ${TRAINABLE_LABEL}
Command: $0 $@
EOF
        ;;

    joint)
        echo -e "${BLUE}Running Joint 5-Fold Cross-Validation (${TRAINABLE_LABEL})${NC}"
        echo ""

        # 创建实验大文件夹（所有 folds 共享）
        EXP_BASE_NAME="h0mini_joint_5fold_${EPOCHS}ep_${TRAINABLE_LABEL}_${DATE}"
        EXP_BASE_DIR="/data/yujk/GLIP/experiments/h0mini_joint/${EXP_BASE_NAME}"
        mkdir -p ${EXP_BASE_DIR}

        # 保存实验总体信息
        cat > ${EXP_BASE_DIR}/experiment_summary.txt << EOF
Experiment: H0-mini Joint 5-Fold Cross-Validation
Date: ${DATE}
Model: H0-mini (${TRAINABLE_LABEL})
Task: Joint Visium + Xenium training
Total Folds: 5
Epochs per Fold: ${EPOCHS}
Batch Size: 32
Visium Samples: 36 BRCA samples
Xenium Sample: NCBI784
Genes: 227 shared genes
Notes: Baseline H0-mini joint experiment, no VAE
Command: $0 $@
EOF

        for FOLD in 0 1 2 3 4; do
            FOLD_DIR="${EXP_BASE_DIR}/fold${FOLD}"

            echo -e "${YELLOW}=========================================="
            echo "Training Fold ${FOLD}/5"
            echo -e "==========================================${NC}"
            echo "Output: ${FOLD_DIR}"
            echo ""

            mkdir -p ${FOLD_DIR}

            ${PYTHON} train_joint_brca_naive.py \
                --model h0mini \
                --run-dir ${FOLD_DIR} \
                --visium-hest-data-dir /data/yujk/hovernet2feature/HEST/hest_data \
                --visium-sample-ids SPA119,SPA120,SPA121,SPA122,SPA123,SPA124,SPA125,SPA126,SPA127,SPA128,SPA129,SPA130,SPA131,SPA132,SPA133,SPA134,SPA135,SPA136,SPA137,SPA138,SPA139,SPA140,SPA141,SPA142,SPA143,SPA144,SPA145,SPA146,SPA147,SPA148,SPA149,SPA150,SPA151,SPA152,SPA153,SPA154 \
                --visium-fold-manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
                --visium-fold-index ${FOLD} \
                --xenium-sample-id NCBI784 \
                --xenium-reference-visium-sample-id SPA124 \
                --pseudo-output-base-dir /data/yujk/GLIP/processed/pseudospots \
                --shared-gene-file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
                --batch-size 32 \
                --epochs ${EPOCHS} \
                --xenium-test-fold 4 \
                --xenium-num-position-folds 5 \
                --max-xenium-train-spots 50000 \
                --max-xenium-test-spots 10000 \
                --image-encoder-checkpoint /data/yujk/H0-mini/pytorch_model.bin \
                --device cuda:0 \
                ${TRAINABLE_FLAG} \
                2>&1 | tee ${FOLD_DIR}/training.log

            # 保存每个 fold 的信息
            cat > ${FOLD_DIR}/fold_info.txt << EOF
Fold: ${FOLD}/5
Epochs: ${EPOCHS}
Status: Completed at $(date)
EOF

            echo -e "${GREEN}Fold ${FOLD} completed!${NC}"
            echo ""
        done

        echo -e "${GREEN}=========================================="
        echo "All 5 folds completed!"
        echo -e "==========================================${NC}"
        echo "Results saved to: ${EXP_BASE_DIR}"

        # 计算 5-fold 汇总统计
        echo ""
        echo -e "${YELLOW}Computing 5-fold summary statistics...${NC}"
        ${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${EXP_BASE_DIR}
        ;;

    *)
        echo -e "${RED}Error: Unknown task '${TASK}'${NC}"
        echo "Valid tasks: visium, xenium, joint"
        exit 1
        ;;
esac

# 记录到全局日志
GLOBAL_LOG="/data/yujk/GLIP/experiments/experiment_log.txt"
echo "$(date): Completed ${TASK} experiment - ${EPOCHS} epochs" >> ${GLOBAL_LOG}

echo ""
echo -e "${GREEN}=========================================="
echo "Experiment Completed!"
echo -e "==========================================${NC}"
echo "Results saved to: /data/yujk/GLIP/experiments/h0mini_${TASK}/"
echo ""
