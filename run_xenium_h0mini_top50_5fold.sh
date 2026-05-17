#!/bin/bash
# H0-mini Xenium pseudo-spot 5-fold, trainable, top_k=50
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BASE="/data/yujk/GLIP/experiments/h0mini_xenium/h0mini_xenium_5fold_20ep_trainable_top50_${DATE}"
PYTHON=/home/yujk/miniconda3/bin/python

mkdir -p ${BASE}
cat > ${BASE}/experiment_info.txt << EOF
Experiment: H0-mini Xenium pseudo-spot 5-fold
Date: ${DATE}
Model: H0-mini (trainable)
top_k: 50
epochs: 20
batch_size: 32
Sample: NCBI784
Reference: SPA124
EOF

for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Fold ${FOLD}/5"
    echo "=========================================="
    mkdir -p ${BASE}/fold${FOLD}
    ${PYTHON} train_xenium_pseudospot.py \
        --model h0mini \
        --trainable \
        --run-dir ${BASE}/fold${FOLD} \
        --sample-id NCBI784 \
        --reference-visium-sample-id SPA124 \
        --pseudo-output-base-dir /data/yujk/GLIP/processed/pseudospots \
        --batch-size 32 \
        --epochs 20 \
        --test-fold ${FOLD} \
        --num-position-folds 5 \
        --top-k 50 \
        --image-encoder-checkpoint /data/yujk/H0-mini/pytorch_model.bin \
        --device cuda:1 \
        2>&1 | tee ${BASE}/fold${FOLD}/training.log
    echo "Fold ${FOLD} done!"
done

echo "All 5 folds done! Results: ${BASE}"
${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${BASE}
