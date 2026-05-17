#!/bin/bash
# ResNet50 Xenium pseudo-spot 5-fold, top_k=50 (baseline)
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BASE="/data/yujk/GLIP/experiments/resnet50_xenium/resnet50_xenium_5fold_20ep_top50_${DATE}"
PYTHON=/home/yujk/miniconda3/bin/python

mkdir -p ${BASE}
cat > ${BASE}/experiment_info.txt << EOF
Experiment: ResNet50 Xenium pseudo-spot 5-fold
Date: ${DATE}
Model: ResNet50 (trainable)
top_k: 50
epochs: 20
batch_size: 64
Sample: NCBI784
Reference: SPA124
EOF

for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Fold ${FOLD}/5"
    echo "=========================================="
    mkdir -p ${BASE}/fold${FOLD}
    ${PYTHON} train_xenium_pseudospot.py \
        --model resnet50 \
        --run-dir ${BASE}/fold${FOLD} \
        --sample-id NCBI784 \
        --reference-visium-sample-id SPA124 \
        --pseudo-output-base-dir /data/yujk/GLIP/processed/pseudospots \
        --batch-size 64 \
        --epochs 20 \
        --test-fold ${FOLD} \
        --num-position-folds 5 \
        --top-k 50 \
        --pretrained false \
        --image-encoder-checkpoint /data/yujk/BLEEP/checkpoints/resnet50_a1_0-14fe96d1.pth \
        --device cuda:0 \
        2>&1 | tee ${BASE}/fold${FOLD}/training.log
    echo "Fold ${FOLD} done!"
done

echo "All 5 folds done! Results: ${BASE}"
${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${BASE}
