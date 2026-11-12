# Notebooks

Grouped by approach. Training code lives in `research/models/`; most current work runs through
`scripts/kaggle_run.py` (any repo module as a Kaggle notebook) rather than hand-written notebooks.

| Folder | What | Status |
|---|---|---|
| `data/` | Dataset builders: `Build_ISLT_LMDB` (ASL Signs → LMDB), `Build_FS_LMDB` (fingerspelling), `preprocess-wlasl-mediapipe` (WLASL videos → landmarks), `build-wlasl-lmdb/` and `build-wlasl-all-lmdb/` (Kaggle kernel folders, push with `kaggle kernels push -p <folder>`); `legacy/` = older LMDB builders | Active |
| `exploration/` | Early data exploration: `data_playground`, `feature_extraction`, `pilot` | Historical |
| `early_transformers/` | First classifier attempts (improved transformer, face-aware transformer, landmark-attention, multi-stream TCN-transformer) — before the LandmarkConformer | Superseded by `islr` approach A |
| `vqvae/` | Approach 0: VQ-VAE pre-training (`sign-vqvae-*`), token classifier, standalone VQ-VAE / translator trainers | Dormant (code: `research/models/vqvae_seq2seq/`) |
| `first_place/` | The published 1st-place Kaggle solution on our signer fold 0 (TFRecords builder + trainer) — an external anchor | Reference |
