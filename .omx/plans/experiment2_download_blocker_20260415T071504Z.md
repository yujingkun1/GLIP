# Experiment 2 download blocker

## Fresh verification evidence
Formal HEST subset download is currently blocked by environment incompatibility:
- current `datasets` version: `4.0.0`
- HEST dataset uses a dataset script (`hest.py`)
- observed error:
  `RuntimeError: Dataset scripts are no longer supported, but found hest.py`

## Official compatibility requirement
HEST official README recommends using:
- `datasets==2.16.0`
- `huggingface-hub==0.20.0`

## Consequence
- Candidate selection is complete
- Mirror metadata access is validated
- Formal sample download must switch to a compatible environment before proceeding

## Formal Experiment 2 targets
- `NCBI681`
- `NCBI682`
- `NCBI683`
- `NCBI684`

## User clarification (BRCA scope)
- The study cancer type is fixed to **BRCA / breast cancer**.
- Current core samples (`SPA119-154`, `NCBI784`) are all on the BRCA line.
- Therefore Experiment 2 should be interpreted as: **same cancer type = BRCA/IDC breast cancer**, **different tissue = non-breast tissue such as lymph node metastasis**.
