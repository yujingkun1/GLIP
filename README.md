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
  --run-dir /data/yujk/GLIP/runs/ncbi784_default \
  --epochs 10 \
  --batch-size 64 \
  --top-k 1
```

## Useful Flags

- `--test-fold`: choose which x-position fold is held out.
- `--num-position-folds`: change the number of contiguous x splits.
- `--remove-control-features false`: keep control features.
- `--nucleus-only true`: only keep transcripts with `overlaps_nucleus == 1`.
- `--force-rebuild-cache`: rebuild the processed cache.
- `--pretrained false`: avoid ImageNet pretrained weights.
- `--final-test-eval-max-cells 0`: run final test Pearson on the full test split.
# GLIP
