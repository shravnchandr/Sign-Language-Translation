# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Isolated Sign Language Recognition system classifying 250 ASL signs from MediaPipe landmarks. The approach: unsupervised Factorized Temporal VQ-VAE pre-training → Conformer-based translator trained on tokenized sequences.

## Commands

```bash
# Install dependencies (UV package manager, Python 3.14)
uv sync

# Train VQ-VAE (Phase 1)
uv run python -m vqvae_seq2seq.vqvae.train_vqvae --data-dir ./data/Isolated_ASL_Recognition --epochs 100

# Train Translator (Phase 2)
uv run python -m vqvae_seq2seq.translation.train_translator --vqvae-checkpoint ./checkpoints/vqvae/best_model.pt

# Standalone scripts (self-contained, no imports from vqvae_seq2seq — use for Kaggle)
uv run python training/train_vqvae.py --data-dir data/Isolated_ASL_Recognition
uv run python training/train_translator.py --data-dir data/Isolated_ASL_Recognition --vqvae-checkpoint checkpoints/best_model.pt

# Format code
uv run black <file_or_directory>
```

## Architecture

### Two-Phase Pipeline

**Phase 1 — Factorized VQ-VAE** (`vqvae_seq2seq/vqvae/`):
- Encodes landmark chunks into 3–4 discrete tokens per chunk: `(pose_id, motion_id, dynamics_id[, face_id])`
- Factorized codebooks: Pose (1024), Motion (512), Dynamics (256), Face (256)
- Multi-scale temporal encoding at chunk sizes `(4, 8, 16)` via `MultiScaleMotionEncoder`
- EMA vector quantization with codebook reset for dead codes
- Cross-factor attention (`CrossFactorAttention`) fuses pose/motion/dynamics representations
- Training is unsupervised — no labels needed; uses all available datasets

**Phase 2 — Sign Translator** (`vqvae_seq2seq/translation/`):
- Input: factorized token sequences from the frozen VQ-VAE
- Encoder: Conformer (CNN + self-attention hybrid) with SpecAugment
- Decoder: Hybrid CTC + Attention decoder (`HybridDecoder`)
- Inference: beam search with CTC prefix scoring (`BeamSearch`)
- 250-class supervised classification using Google ASL Signs labels

### Data Flow
```
Parquet file → LandmarkProcessor → (T, N, 3) tensor
  → VQ-VAE → [(pose_id, motion_id, dyn_id), ...] per chunk
  → FactorizedTokenEmbedding → Conformer → 250-class output
```

### Key Modules
| File | Purpose |
|------|---------|
| `vqvae_seq2seq/vqvae/config.py` | `ImprovedVQVAEConfig` — all hyperparameters |
| `vqvae_seq2seq/vqvae/vqvae_model.py` | Main VQ-VAE model (assembles all sub-modules) |
| `vqvae_seq2seq/vqvae/vector_quantizer.py` | `EMAVectorQuantizer`, `FactorizedVectorQuantizer`, `DiversityLoss` |
| `vqvae_seq2seq/vqvae/multi_scale_encoder.py` | Multi-scale motion encoding |
| `vqvae_seq2seq/vqvae/face_encoder.py` | Dedicated face NMM encoder (5 regions) |
| `vqvae_seq2seq/vqvae/hand_dominance.py` | Detects & reorders dominant/non-dominant hands |
| `vqvae_seq2seq/data/preprocessing.py` | `RobustPreprocessor`, `LandmarkConfig` |
| `vqvae_seq2seq/data/dataset.py` | `VQVAEDataset`, `TranslationDataset` |
| `vqvae_seq2seq/translation/translator_model.py` | `SignTranslator` (full model) |
| `vqvae_seq2seq/translation/conformer.py` | Conformer encoder blocks |
| `vqvae_seq2seq/translation/config.py` | `TranslationConfig` |
| `training/train_vqvae.py` | Standalone VQ-VAE training (duplicates vqvae_seq2seq for Kaggle) |
| `training/train_translator.py` | Standalone translator training |

## Data

### Datasets (under `data/`)
- `data/Isolated_ASL_Recognition/` — Google ASL Signs (94k videos, 250 signs) — primary labeled dataset
- `data/ASL_Fingerspelling_Recognition/` — Large fingerspelling dataset (unlabeled pre-training)
- `data/WLASL_Landmarks/` — WLASL landmarks after MediaPipe preprocessing

### Parquet Format
Columns: `frame`, `type`, `landmark_index`, `x`, `y`, `z`
- `type`: `'pose'` (33), `'left_hand'` (21), `'right_hand'` (21), `'face'` (478) — 553 total per frame
- Coordinates normalized to [0, 1] by MediaPipe; further normalized body-relative at runtime

### Download
```bash
kaggle competitions download -c asl-signs               # Google ASL Signs
kaggle competitions download -c asl-fingerspelling       # Fingerspelling
```

## Key Patterns

**Normalization fallback chain** (`RobustPreprocessor`): nose → shoulder center → hip center. Subtracts the origin to make coordinates body-relative. Falls back when nose landmark is missing.

**Hand dominance** (`HandDominanceModule`): Detects dominant hand from wrist velocity; reorders left/right channels so dominant hand is always in the first slot before encoding.

**Variable-length batching**: All datasets return a `padding_mask` `(B, T)` bool tensor (`True` = padding). Pass to model alongside `landmarks`.

**AMP training**: Use `torch.amp.autocast` + `GradScaler` (already in standalone scripts). Config has `device: str = "cuda"` — override to `"mps"` or `"cpu"` locally.

**Standalone vs modular**: `training/` scripts inline all code from `vqvae_seq2seq/` to be self-contained for Kaggle notebook submission. When editing model logic, update both `vqvae_seq2seq/` and `training/` if the change needs to run on Kaggle.

**Label mapping**: 250 ASL signs indexed 0–249. Mapping lives in `data/Isolated_ASL_Recognition/sign_to_prediction_index_map.json`.
