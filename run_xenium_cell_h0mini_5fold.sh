#!/bin/bash
# H0-mini Xenium cell-level 5-fold
# 对应: brca_stage4_cell_xenium_only_resnet50_g227_5fold
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BASE="/data/yujk/GLIP/experiments/h0mini_xenium/h0mini_xenium_cell_5fold_20ep_trainable_${DATE}"
PYTHON=/home/yujk/miniconda3/bin/python

mkdir -p ${BASE}

for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Fold ${FOLD}/5"
    echo "=========================================="
    mkdir -p ${BASE}/fold${FOLD}
    ${PYTHON} train_xenium.py \
        --model h0mini \
        --trainable \
        --run-dir ${BASE}/fold${FOLD} \
        --sample-id NCBI784 \
        --gene-file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
        --batch-size 64 \
        --epochs 20 \
        --test-fold ${FOLD} \
        --num-position-folds 5 \
        --top-k 1 \
        --image-encoder-checkpoint /data/yujk/H0-mini/pytorch_model.bin \
        --device cuda:0 \
        2>&1 | tee ${BASE}/fold${FOLD}/training.log
    echo "Fold ${FOLD} done!"
done

echo "All 5 folds done! Results: ${BASE}"
