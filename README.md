# GLIP

Single-cell contrastive training on HEST Xenium `NCBI784`.

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

## Train

```bash
python /data/yujk/GLIP/train.py \
  --run-dir /data/yujk/GLIP/runs/ncbi784_uni \
  --epochs 10 \
  --batch-size 64 \
  --top-k 1
```

The default training entrypoint now starts from the local UNI2-h checkpoint at
`/data/yujk/UNI2-h/pytorch_model.bin`. If you need to specify it explicitly:

```bash
python /data/yujk/GLIP/train.py \
  --run-dir /data/yujk/GLIP/runs/ncbi784_uni \
  --epochs 10 \
  --batch-size 64 \
  --model uni \
  --image-encoder-checkpoint /data/yujk/UNI2-h/pytorch_model.bin \
  --top-k 1
```

If direct access to `huggingface.co` is slow from your network, you can route
Hub downloads through a mirror endpoint and extend the timeout:

```bash
python /data/yujk/GLIP/train.py \
  --run-dir /data/yujk/GLIP/runs/ncbi784_uni \
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
python /data/yujk/GLIP/train.py \
  --run-dir /data/yujk/GLIP/runs/ncbi784_uni_local \
  --epochs 10 \
  --batch-size 64 \
  --model uni \
  --pretrained false \
  --image-encoder-checkpoint /path/to/pytorch_model.bin \
  --top-k 1
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
