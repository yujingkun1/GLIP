# GLIP

Unified workspace for:

- Xenium single-cell training
- Xenium pseudo-spot training
- Visium/ST spot-level BLEEP training

The original `BLEEP` Visium pipeline has been merged into this repository
under `glip/visium/` so Xenium- and Visium-level models can be managed in one
codebase. Core training logic is kept unchanged; only the code organization and
default output directories were cleaned up.

## What This Project Does

- Builds a cell-level cache directly from:
  - `xenium_seg/NCBI784_xenium_cell_seg.parquet`
  - `transcripts/NCBI784_transcripts.parquet`
  - `wsis/NCBI784.tif`
- Uses one H&E cell-centered image crop per cell and the matched cell-level gene expression vector as supervision.
- Keeps the BLEEP-style setup:
  - image encoder
  - gene projection head
  - symmetric contrastive loss
  - retrieval-based Pearson evaluation
- Splits train/test by contiguous x-position folds on the image.

## Default Data Assumptions

- Sample id: `NCBI784`
- Split rule: 5 contiguous x-position folds, `test_fold=4`
- Default preprocessing:
  - remove `BLANK`, `NegControl`, `antisense`, `Unassigned`
  - keep all assigned transcripts, not only `overlaps_nucleus == 1`
  - drop segmented cells with zero remaining transcripts

## Prepare Cache

```bash
python /data/yujk/GLIP/prepare_ncbi784.py
```

## Entry Points

- Xenium cell-level:
  - `python /data/yujk/GLIP/train_xenium.py`
  - legacy entrypoint: `python /data/yujk/GLIP/train.py`
- Xenium pseudo-spot:
  - `python /data/yujk/GLIP/train_xenium_pseudospot.py`
  - legacy entrypoint: `python /data/yujk/GLIP/train_pseudospot.py`
- Visium/ST spot-level:
  - `python /data/yujk/GLIP/train_visium.py`

## Output Layout

- Xenium-family runs default to `/data/yujk/GLIP/runs_xenium`
- Visium-family runs default to `/data/yujk/GLIP/runs_visium`

## Train Xenium Cell-Level

```bash
python /data/yujk/GLIP/train_xenium.py \
  --run-dir /data/yujk/GLIP/runs_xenium/ncbi784_uni \
  --epochs 10 \
  --batch-size 64 \
  --top-k 1
```

The default training entrypoint now starts from the local UNI2-h checkpoint at
`/data/yujk/UNI2-h/pytorch_model.bin`. If you need to specify it explicitly:

```bash
python /data/yujk/GLIP/train_xenium.py \
  --run-dir /data/yujk/GLIP/runs_xenium/ncbi784_uni \
  --epochs 10 \
  --batch-size 64 \
  --model uni \
  --image-encoder-checkpoint /data/yujk/UNI2-h/pytorch_model.bin \
  --top-k 1
```

If direct access to `huggingface.co` is slow from your network, you can route
Hub downloads through a mirror endpoint and extend the timeout:

```bash
python /data/yujk/GLIP/train_xenium.py \
  --run-dir /data/yujk/GLIP/runs_xenium/ncbi784_uni \
  --epochs 10 \
  --batch-size 64 \
  --model uni \
  --hf-endpoint https://hf-mirror.com \
  --hf-hub-download-timeout 120 \
  --hf-hub-etag-timeout 120 \
  --top-k 1
```

If you already downloaded the UNI2-h checkpoint locally, you can avoid Hub
access entirely:

```bash
python /data/yujk/GLIP/train_xenium.py \
  --run-dir /data/yujk/GLIP/runs_xenium/ncbi784_uni_local \
  --epochs 10 \
  --batch-size 64 \
  --model uni \
  --pretrained false \
  --image-encoder-checkpoint /path/to/pytorch_model.bin \
  --top-k 1
```

## Train Xenium Pseudo-Spot

```bash
python /data/yujk/GLIP/train_xenium_pseudospot.py \
  --run-dir /data/yujk/GLIP/runs_xenium/ncbi784_pseudospot \
  --reference-visium-sample-id SPA124 \
  --epochs 20 \
  --batch-size 64
```

## Train Visium/ST Spot-Level

```bash
python /data/yujk/GLIP/train_visium.py \
  --exp_name /data/yujk/GLIP/runs_visium/spa_loo_resnet50 \
  --hest_data_dir /data/yujk/hovernet2feature/HEST/hest_data \
  --gene_file /data/yujk/hovernet2feature/HisToGene/data/her_hvg_cut_1000.txt \
  --batch_size 32 \
  --max_epochs 10 \
  --model resnet50 \
  --pretrained false \
  --image_encoder_checkpoint /data/yujk/BLEEP/checkpoints/resnet50_a1_0-14fe96d1.pth \
  --device_id 0 \
  --num_workers 0
```

## Useful Flags

- `--model resnet50|uni`: choose the image encoder.
- `--test-fold`: choose which x-position fold is held out.
- `--num-position-folds`: change the number of contiguous x splits.
- `--remove-control-features false`: keep control features.
- `--nucleus-only true`: only keep transcripts with `overlaps_nucleus == 1`.
- `--force-rebuild-cache`: rebuild the processed cache.
- `--pretrained false`: avoid remote pretrained weights and use random/local initialization.
- `--image-encoder-checkpoint`: load a local image encoder checkpoint.
- `--hf-endpoint`: route HF Hub downloads through a mirror such as `https://hf-mirror.com`.
- `--final-test-eval-max-cells 0`: run final test Pearson on the full test split.
