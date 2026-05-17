#!/bin/bash
# H0-mini Visium 5-fold, trainable, top_k=50, anti-overfitting params
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BASE="/data/yujk/GLIP/experiments/h0mini_visium/h0mini_visium_5fold_15ep_trainable_wd1e-2_lr5e-5_${DATE}"
PYTHON=/home/yujk/miniconda3/bin/python

mkdir -p ${BASE}
cat > ${BASE}/experiment_info.txt << EOF
Experiment: H0-mini Visium 5-fold (anti-overfitting)
Date: ${DATE}
Model: H0-mini (trainable)
top_k: 50
epochs: 15
batch_size: 32
lr: 5e-5
weight_decay: 1e-2
Data: BRCA Visium (36 samples)
Genes: 227
EOF

for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Fold ${FOLD}/5"
    echo "=========================================="
    mkdir -p ${BASE}/fold${FOLD}
    ${PYTHON} train_visium.py \
        --model h0mini \
        --trainable \
        --exp_name ${BASE}/fold${FOLD} \
        --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
        --gene_file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
        --batch_size 32 \
        --max_epochs 15 \
        --num_workers 8 \
        --top_k 50 \
        --lr 1e-5 \
        --weight_decay 1e-2 \
        --cv_mode fixed_manifest \
        --fold_manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
        --fold_index ${FOLD} \
        --pretrained true \
        --image_encoder_checkpoint /data/yujk/H0-mini/pytorch_model.bin \
        --device_id 1 \
        2>&1 | tee ${BASE}/fold${FOLD}/training.log
    echo "Fold ${FOLD} done!"
done

echo "All 5 folds done! Results: ${BASE}"
${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${BASE}
