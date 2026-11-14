# Retired data notebooks

Do not run these to build training or evaluation data. They predate the data contract
(`research/models/islr/core/data/contract.py`, cache version `v3_contract`):

- `build_asl_signs_lmdb.ipynb`, `create-is-lmdb.ipynb`: fill missing landmarks before recording
  detection and store coordinates only (no presence, no provenance) — the current loaders reject
  their output.
- `preprocess-wlasl-mediapipe.ipynb`: one MediaPipe tracker per worker shared across videos,
  version not pinned (replaced by `scripts/extract_wlasl.py`).
- `build-wlasl-lmdb/`, `build-wlasl-all-lmdb/`: build from that extraction with `--wlasl-dir`; the
  selection is not recorded, so `validate.py` fails their output.
- `create-fs-lmdb.ipynb`: older fingerspelling builder.

Current builds: `scripts/kaggle_run.py push ... -- islr.core.data.build_* ... :: islr.core.data.validate --raw-root ...`
(see `docs/experiments/05_data_pipeline_audit.md`).
