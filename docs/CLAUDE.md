# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Isolated Sign Language Recognition system classifying 250 ASL signs from MediaPipe landmarks. The approach: unsupervised Factorized Temporal VQ-VAE pre-training → Conformer-based translator trained on tokenized sequences.

## Commands

```bash
# Install dependencies (UV package manager, Python 3.14)
uv sync

# Full pipeline (Phase 1 → pre-tokenize → Phase 2)
bash run_pipeline.sh
bash run_pipeline.sh --vqvae-epochs 10 --translator-epochs 10  # quick test

# Train VQ-VAE (Phase 1)
uv run python -m vqvae_seq2seq.vqvae.train_vqvae --data-dir data/Isolated_ASL_Recognition --cache-dir data/cache --epochs 100

# Pre-tokenize dataset with trained VQ-VAE (run once after Phase 1)
uv run python -m vqvae_seq2seq.scripts.precompute_tokens --vqvae-checkpoint checkpoints/vqvae/best_model.pt --data-dir data/Isolated_ASL_Recognition --token-dir data/tokens --cache-dir data/cache --num-workers 4

# Train Translator (Phase 2) — fast path using pre-tokenized data
uv run python -m vqvae_seq2seq.translation.train_translator --token-dir data/tokens --data-dir data/Isolated_ASL_Recognition --epochs 100

# Format code
uv run black <file_or_directory>
```

## Architecture

### Two-Phase Pipeline

**Phase 1 — Factorized VQ-VAE** (`vqvae_seq2seq/vqvae/`):
- Encodes landmark chunks into 4 discrete tokens per chunk: `(pose_id, motion_id, dynamics_id, face_id)`
- Factorized codebooks: Pose (256), Motion (256), Dynamics (128), Face (128)
- Multi-scale temporal encoding at chunk sizes `(4, 8, 16)` via `MultiScaleMotionEncoder`
- EMA vector quantization with soft diversity loss and codebook reset for dead codes
- Cross-factor attention (`CrossFactorAttention`) fuses pose/motion/dynamics representations
- Training is unsupervised — no labels needed; uses all available datasets

**Phase 2 — Sign Translator** (`vqvae_seq2seq/translation/`):
- Input: pre-tokenized indices loaded from `data/tokens/` (no VQ-VAE in memory during training)
- Encoder: Conformer (CNN + self-attention, kernel=7, 6 layers, d_model=256)
- Decoder: Hybrid CTC + Attention decoder (`HybridDecoder`, 3 layers)
- Inference: beam search with CTC prefix scoring (`BeamSearch`)
- 250-class supervised classification using Google ASL Signs labels

### Data Flow
```
Parquet file → LandmarkProcessor → (T, N, 3) tensor
  → VQ-VAE → [(pose_id, motion_id, dyn_id, face_id), ...] per chunk  [saved to data/tokens/]
  → FactorizedTokenEmbedding → Conformer (kernel=7) → 250-class output
```

### Key Modules
| File | Purpose |
|------|---------|
| `vqvae_seq2seq/vqvae/config.py` | `ImprovedVQVAEConfig` — all hyperparameters |
| `vqvae_seq2seq/vqvae/vqvae_model.py` | Main VQ-VAE model (assembles all sub-modules) |
| `vqvae_seq2seq/vqvae/vector_quantizer.py` | `EMAVectorQuantizer`, `FactorizedVectorQuantizer` |
| `vqvae_seq2seq/vqvae/multi_scale_encoder.py` | Multi-scale motion encoding |
| `vqvae_seq2seq/vqvae/face_encoder.py` | Dedicated face NMM encoder (5 regions) |
| `vqvae_seq2seq/vqvae/hand_dominance.py` | Detects & reorders dominant/non-dominant hands |
| `vqvae_seq2seq/scripts/precompute_tokens.py` | Pre-tokenize dataset with frozen VQ-VAE |
| `vqvae_seq2seq/data/preprocessing.py` | `RobustPreprocessor`, `LandmarkConfig` |
| `vqvae_seq2seq/data/dataset.py` | `VQVAEDataset`, `TranslationDataset`, `TokenizedTranslationDataset` |
| `vqvae_seq2seq/translation/translator_model.py` | `SignTranslator` (full model) |
| `vqvae_seq2seq/translation/conformer.py` | Conformer encoder blocks |
| `vqvae_seq2seq/translation/config.py` | `TranslationConfig` |
| `run_pipeline.sh` | End-to-end pipeline script |

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

**Soft diversity loss** (`EMAVectorQuantizer`): computed from the distance matrix using `softmax(-distances)` before the argmin step. Gradients flow through `z_flat` to the encoder, pushing it toward spread-out representations. The codebook (EMA buffer) is detached — only the encoder receives this gradient.

**Variable-length batching**: All datasets return a `padding_mask` `(B, T)` bool tensor (`True` = valid). Pass to model alongside `landmarks` or token indices.

**AMP training**: `torch.amp.autocast` + `GradScaler` wrap every forward/backward pass in `train_vqvae.py`. Disabled automatically when not on CUDA.

**Preprocessing cache**: `VQVAEDataset` accepts `cache_dir`. First access processes each parquet and saves a `.pt` tensor; subsequent accesses skip parquet parsing entirely. Default: `data/cache/`.

**Pre-tokenization cache**: `precompute_tokens.py` runs the frozen VQ-VAE once, saves per-sample token indices to `data/tokens/` mirroring the data directory structure. `TokenizedTranslationDataset` loads these directly — no VQ-VAE needed during Phase 2 training.

**Conformer kernel size**: must be significantly smaller than the average sequence length. VQ-VAE chunk size 8 → 40–80 frame signs produce 5–10 tokens. `encoder_kernel_size=7` fits within the sequence; larger values operate mostly on padding.

**Best model selection**: Phase 1 saves checkpoints based on val reconstruction loss (not total loss). The diversity term dominates total loss magnitude and is a poor ranking signal.

**Standalone vs modular**: `training/` scripts inline all code from `vqvae_seq2seq/` to be self-contained for Kaggle notebook submission. When editing model logic, update both `vqvae_seq2seq/` and `training/` if the change needs to run on Kaggle.

**Label mapping**: 250 ASL signs indexed 0–249. Mapping lives in `data/Isolated_ASL_Recognition/sign_to_prediction_index_map.json`.
