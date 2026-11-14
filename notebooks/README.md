# Notebooks

Grouped by approach. Training code lives in `research/models/`; most current work runs through
`scripts/kaggle_run.py` (any repo module as a Kaggle notebook) rather than hand-written notebooks.

| Folder | What | Status |
|---|---|---|
| `data/` | `build_fingerspelling_lmdb` (fingerspelling LMDB; pre-contract format, CTC pre-training only). **Sign LMDBs are no longer built by notebooks**: contract-v3 builds run as `scripts/kaggle_run.py` notebooks (`islr.core.data.build_*` then `islr.core.data.validate --raw-root`; commands in `docs/experiments/05_data_pipeline_audit.md`), landmarks from `scripts/extract_*.py`. `legacy/` = retired builders that write pre-contract LMDBs the loaders reject (`build_asl_signs_lmdb`, `build-wlasl-lmdb/`, `build-wlasl-all-lmdb/`, `preprocess-wlasl-mediapipe` — shared tracker, unpinned MediaPipe) and older `create-*-lmdb` | Active |
| `exploration/` | Early data exploration: `data_playground`, `feature_extraction`, `pilot` | Historical |
| `early_transformers/` | First classifier attempts (improved transformer, face-aware transformer, landmark-attention, multi-stream TCN-transformer) — before the LandmarkConformer | Superseded by `islr` approach A |
| `vqvae/` | Approach 0: VQ-VAE pre-training (`sign-vqvae-*`), token classifier, standalone VQ-VAE / translator trainers | Dormant (code: `research/models/vqvae_seq2seq/`) |
| `first_place/` | The published 1st-place Kaggle solution on our signer fold 0 (TFRecords builder + trainer) — an external anchor | Reference |
