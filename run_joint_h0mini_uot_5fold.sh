#!/bin/bash
# H0-mini Joint UOT (image_ot_targetbank_full, uot rho=2.0), 5-fold
# 对应: brca_stage2_image_ot_targetbank_full_resnet50_g227_random5fold_bs64_uot_rho2p0
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BASE="/data/yujk/GLIP/experiments/h0mini_joint/h0mini_joint_uot_5fold_50ep_${DATE}"
PYTHON=/home/yujk/miniconda3/bin/python

mkdir -p ${BASE}

for FOLD in 0 1 2 3 4; do
    echo "=========================================="
    echo "Fold ${FOLD}/5"
    echo "=========================================="
    mkdir -p ${BASE}/fold${FOLD}
    ${PYTHON} train_joint_brca_naive.py \
        --model h0mini \
        --trainable \
        --run-dir ${BASE}/fold${FOLD} \
        --visium-hest-data-dir /data/yujk/hovernet2feature/HEST/hest_data \
        --visium-sample-ids SPA119,SPA120,SPA121,SPA122,SPA123,SPA124,SPA125,SPA126,SPA127,SPA128,SPA129,SPA130,SPA131,SPA132,SPA133,SPA134,SPA135,SPA136,SPA137,SPA138,SPA139,SPA140,SPA141,SPA142,SPA143,SPA144,SPA145,SPA146,SPA147,SPA148,SPA149,SPA150,SPA151,SPA152,SPA153,SPA154 \
        --visium-fold-manifest /data/yujk/GLIP/configs/brca_visium_random5fold_seed42.json \
        --visium-fold-index ${FOLD} \
        --xenium-sample-id NCBI784 \
        --xenium-reference-visium-sample-id SPA124 \
        --pseudo-output-base-dir /data/yujk/GLIP/processed/pseudospots \
        --shared-gene-file /data/yujk/GLIP/configs/brca_shared_genes_ncbi784_visium36_intersection_227.txt \
        --batch-size 32 \
        --epochs 50 \
        --lr 1e-5 \
        --weight-decay 1e-3 \
        --top-k 50 \
        --eval-bank-mode target \
        --module-image-ot true \
        --ot-transport uot \
        --ot-image-weight 0.05 \
        --ot-gene-weight 0.05 \
        --ot-sinkhorn-eps 0.05 \
        --ot-sinkhorn-iters 50 \
        --uot-marginal-weight 2.0 \
        --module-naive-joint true \
        --image-encoder-checkpoint /data/yujk/H0-mini/pytorch_model.bin \
        --device cuda:0 \
        2>&1 | tee ${BASE}/fold${FOLD}/training.log
    echo "Fold ${FOLD} done!"
done

echo "All 5 folds done! Results: ${BASE}"
${PYTHON} /data/yujk/GLIP/compute_fold_summary.py ${BASE}
