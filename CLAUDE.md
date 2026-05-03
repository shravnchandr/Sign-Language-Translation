# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Isolated Sign Language Recognition system classifying 250 ASL signs from MediaPipe landmarks.

Two approaches are under active development, both under `research/models/`:

| Approach | Location | Status |
|---|---|---|
| Factorized VQ-VAE → Conformer translator | `research/models/vqvae_seq2seq/` | Primary pipeline |
| AnatomicalConformer (end-to-end) | `research/models/cnn_transformer/` | Kaggle training target |
| ST-GCN (baseline) | `research/models/st_gcn/` | Experimental |

## Commands

```bash
# Install dependencies (UV package manager, Python 3.14)
uv sync

# Full pipeline (run from project root — requires research/models/ on PYTHONPATH)
PYTHONPATH=research/models bash run_pipeline.sh
PYTHONPATH=research/models bash run_pipeline.sh --vqvae-epochs 10 --translator-epochs 10

# Train VQ-VAE (Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition --cache-dir data/cache --epochs 100

# Pre-tokenize dataset with trained VQ-VAE (run once after Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition --token-dir data/tokens \
  --cache-dir data/cache --num-workers 4

# Train Translator (Phase 2) — fast path using pre-tokenized data
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens --data-dir data/Isolated_ASL_Recognition --epochs 100

```

## Architecture

### Approach 1 — Factorized VQ-VAE Pipeline (`research/models/vqvae_seq2seq/`)

**Phase 1 — Factorized VQ-VAE:**
- Encodes landmark chunks into 4 discrete tokens per chunk: `(pose_id, motion_id, dynamics_id, face_id)`
- Factorized codebooks: Pose (256), Motion (256), Dynamics (128), Face (128)
- Multi-scale temporal encoding at chunk sizes `(4, 8, 16)` via `MultiScaleMotionEncoder`
- EMA vector quantization with soft diversity loss and codebook reset for dead codes
- Cross-factor attention (`CrossFactorAttention`) fuses pose/motion/dynamics representations
- `HandDominanceModule` reorders left/right hands so dominant hand is always in the first slot
- Training is unsupervised — no labels needed; uses all available datasets

**Phase 2 — Sign Translator:**
- Input: pre-tokenized indices loaded from `data/tokens/` (no VQ-VAE in memory during training)
- Encoder: Conformer (CNN + self-attention, kernel=7, 6 layers, d_model=256)
- Decoder: Hybrid CTC + Attention decoder (`HybridDecoder`, 3 layers)
- Inference: beam search with CTC prefix scoring (`BeamSearch`)
- 250-class supervised classification using Google ASL Signs labels

**Data Flow:**
```
Parquet → LandmarkProcessor → (T, N, 3)
  → RobustPreprocessor → HandDominanceModule
  → MultiScaleMotionEncoder → CrossFactorAttention
  → FactorizedVectorQuantizer → [(pose_id, motion_id, dyn_id, face_id), ...]  [saved to data/tokens/]
  → FactorizedTokenEmbedding → Conformer → HybridDecoder → 250-class output
```

### Approach 2 — AnatomicalConformer (`research/models/cnn_transformer/`)

End-to-end supervised classification. Optional CTC pre-training on ASL Fingerspelling initialises the backbone before fine-tuning on the 250-class task. Designed for Kaggle training.

- Input coordinates: x, y, z per landmark (`INCLUDE_DEPTH=True`)
- Per-body-part projection: separate `nn.Linear` for LH, RH, pose; face is split into `eyebrow_proj` (grammatical: questions/negation) and `mouth_proj` (phonological: mouthing), each at `d_model//8` — same total budget as a single face projection
- Multi-scale velocity stream (Δ1/Δ2/Δ5): Δ1 computed in dataset from raw coords; Δ2 and Δ5 computed inside `forward()` from body-relative positions (after nose subtraction). All three scales concatenated per part and projected to `d_model//4`. Position and velocity get equal `d_model` budget before fusion.
- Geometry stream: per hand, 15 joint-angle cosines (3 per finger, at MCP/PIP/DIP joints) + 10 fingertip pairwise distances = 25 features/hand. Computed in `_hand_geometry()` from wrist-relative fingers after `WristNormalization`. Projected 2 × `d_model//8` = `d_model//4`. Invariant to wrist rotation and signer hand scale — encodes fine-grained hand shape that raw XYZ obscures.
- Feature fusion: pos (`d_model`) + vel (`d_model`) + geo (`d_model//4`) → `feat_fuse` → `d_model`
- Conformer blocks (depthwise conv + self-attention) + CLS token for classification
- Body-relative normalization done once at LMDB build time (`normalize_values`: nose → shoulder → hip → 0 fallback). `WristNormalization` applied in-model: landmark 0 = location (nose-relative), landmarks 1–20 = shape (wrist-relative).
- **Optional fingerspelling CTC pre-training** (`pretrain_fingerspelling.py`): trains the backbone on `ASL_Fingerspelling_Recognition` using `nn.CTCLoss` (60-token char vocab + blank). CTC mode skips the CLS token and returns per-frame logits `(B, T, vocab+1)`. Signer-independent split via `GroupShuffleSplit` on `participant_id`. Saves `backbone_best.pth` (all keys except `head.`, `ctc_head.`, `signer_disc.`, `cls_token`). Loaded at fine-tuning start via `--pretrained-backbone` with `strict=False`.
- Single-phase training: Phase 1 (100 epochs default, heavy aug, mixup, OneCycleLR)
- Test-time augmentation (5-pass TTA) at evaluation
- Stochastic depth (`drop_path_max=0.1`): linearly increasing per-block skip probability (block 0 = 0, last = drop_path_max). Controlled via `--drop-path-max` CLI arg.
- GRL signer-invariance (`--grl-lambda 0.1`): `SignerDiscriminator` on CLS token, gradient reversed so feature extractor is forced to discard signer identity. Ganin schedule ramps λ from 0 → max over Phase 1; Phase 2 continues the ramp from Phase 1's endpoint (stays at ~max_lambda). Requires `participant_id` in train.csv; auto-disables when not present. Adversarial loss uses same mixup weighting as sign loss. Discriminator accuracy is logged each epoch alongside chance level (`1/n_signers`) to verify the feature extractor is successfully confusing the discriminator.

## Key Modules

### VQ-VAE pipeline (`research/models/vqvae_seq2seq/`)

| File | Purpose |
|------|---------|
| `vqvae/config.py` | `ImprovedVQVAEConfig` — all hyperparameters |
| `vqvae/vqvae_model.py` | Main VQ-VAE model (assembles all sub-modules) |
| `vqvae/vector_quantizer.py` | `EMAVectorQuantizer`, `FactorizedVectorQuantizer` |
| `vqvae/multi_scale_encoder.py` | Multi-scale motion encoding |
| `vqvae/face_encoder.py` | Dedicated face NMM encoder (5 regions) |
| `vqvae/hand_dominance.py` | Detects & reorders dominant/non-dominant hands |
| `vqvae/cross_attention.py` | `CrossFactorAttention` fuses pose/motion/dynamics |
| `scripts/precompute_tokens.py` | Pre-tokenize dataset with frozen VQ-VAE |
| `data/preprocessing.py` | `RobustPreprocessor`, `LandmarkProcessor` |
| `data/dataset.py` | `VQVAEDataset`, `TranslationDataset`, `TokenizedTranslationDataset` |
| `translation/translator_model.py` | `SignTranslator` (full model) |
| `translation/conformer.py` | Conformer encoder blocks |
| `translation/decoder.py` | `HybridDecoder` (CTC + attention) |
| `translation/beam_search.py` | Beam search with CTC prefix scoring |
| `translation/config.py` | `TranslationConfig` |

### AnatomicalConformer (`research/models/cnn_transformer/`)

| File | Purpose |
|------|---------|
| `config.py` | Landmark layout constants, feature dimensions |
| `model/anatomical_conformer.py` | `AnatomicalConformer` — main model |
| `model/conformer.py` | `ConformerBlock`, `SinusoidalPositionalEncoding` |
| `model/normalization.py` | `WristNormalization` |
| `model/grl.py` | `SignerDiscriminator`, `ganin_lambda` — GRL signer-invariance |
| `data/dataset.py` | `ASLDataset`, `BucketBatchSampler`, `get_data_loaders` |
| `data/augmentation.py` | `AdvancedAugmentation` (7 types), `mixup_batch` |
| `data/preprocessing.py` | `frame_stacked_data` — parquet → numpy array |
| `data/build_lmdb.py` | One-time LMDB archive builder (parallelised — `os.cpu_count()` workers by default) |
| `data/build_fingerspelling_lmdb.py` | Fingerspelling LMDB builder — parquet-file-level parallelism (one worker per ~1 GB file) |
| `data/fingerspelling_dataset.py` | `FingerspellingDataset`, `collate_ctc`, `load_char_map` — CTC pre-training data pipeline |
| `data/_cache_keys.py` | `CACHE_VERSION` hash, `lmdb_key`/`lmdb_length_key` helpers |
| `pretrain_fingerspelling.py` | CTC pre-training loop — saves `backbone_best.pth` for fine-tuning |
| `train.py` | Two-phase training loop with TTA evaluation |

## Known Bugs

| File | Line | Issue |
|------|------|-------|
| `research/models/vqvae_seq2seq/vqvae/vqvae_model.py` | 348–353 | `decode()` maps wrong factors to slots: pose_q→dominant_hand, motion_q→non_dominant_hand, dynamics_q→pose. Should match each factor to its own slot. |
| `research/models/vqvae_seq2seq/vqvae/vector_quantizer.py` | 201 | MSE loss arguments reversed: `F.mse_loss(z_q.detach(), z)` should be `F.mse_loss(z, z_q.detach())` to pull input toward codebook. |
| `research/models/vqvae_seq2seq/vqvae/hand_dominance.py` | 173–193 | `HandMirrorAugmentation` doesn't correctly flip x-coordinates. |
| `research/models/vqvae_seq2seq/translation/train_translator.py` | 117 | Chunk size hardcoded as `8` when computing encoder lengths. Should use `config.base_chunk_size`. |
| `research/models/st_gcn/st_gcn_model.py` | 106–110 | `edge_importance` parameter is allocated but never used in `forward()`. |
| `research/models/st_gcn/st_gcn_training.py` | 219–225 | Double-normalizes adjacency matrix: `LandmarkGraph.get_normalized_adjacency()` already normalizes, then chain-edges are added and it's normalized again. |
| `research/models/cnn_transformer/data/preprocessing.py` | — | ~~BASE_PATH double-prefixing: `frame_stacked_data` prepended `BASE_PATH` to an already-absolute path built by `dataset.py`. Fixed: `pd.read_parquet(file_path)` directly.~~ **Fixed.** |
| `research/models/cnn_transformer/model/anatomical_conformer.py` | — | ~~Hand dominance swap inverted: `lh_energy > rh_energy` triggered swap when LH was already dominant.~~ **Fixed: `rh_energy > lh_energy`.** |
| `research/models/cnn_transformer/model/normalization.py` | — | ~~`RobustNormalization` mutated input in-place, corrupting TTA source tensors.~~ **Fixed: clones output before writing.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`random_flip` double-negated velocity x-coords (two loop passes, each covering full tensor).~~ **Fixed: single pass.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`mixup_batch` discarded the shuffled sample's mask.~~ **Fixed: returns `mask \| mask[index]`.** |
| `research/models/cnn_transformer/train.py` | — | ~~OneCycleLR `steps_per_epoch=len(train_loader)` ignored gradient accumulation, making schedule 2–4× slower.~~ **Fixed: `total_steps` computed from actual optimizer step counts per phase.** |
| `research/models/cnn_transformer/train.py` | — | ~~Phase 2 loop `range(epoch_idx+1, total_steps)` ran too many epochs after early stopping.~~ **Fixed: `range(NUM_EPOCHS_PHASE2)`.** |
| `research/models/cnn_transformer/train.py` | — | ~~Validation used 5× stochastic TTA, making checkpoint selection noisy.~~ **Fixed: `evaluate_epoch` is deterministic; TTA reserved for final reporting via `evaluate_epoch_tta`.** |
| `research/models/cnn_transformer/model/conformer.py` | — | ~~Depthwise conv ran over padded positions, leaking zeros into valid boundary frames.~~ **Fixed: conv residual zeroed at padded positions.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`mixup_batch` paired lh-dominant with rh-dominant samples, producing ambiguous hand slot assignments when `HandDominanceModule` runs inside the model.~~ **Fixed: dominance-aware pairing shuffles within same-dominance groups.** |
| `research/models/cnn_transformer/data/preprocessing.py` + `model/anatomical_conformer.py` | — | ~~Double normalization: `normalize_values` zeroed the nose before LMDB, so `RobustNormalization` in the model always fell through to the shoulder-center branch (nose appeared missing).~~ **Fixed: `normalize_values` implements the full nose→shoulder→hip fallback; `RobustNormalization` removed from model.** LMDB must be rebuilt (`_NORM_VERSION` bump auto-invalidates). |

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

**Normalization fallback chain** (`RobustPreprocessor`, `RobustNormalization`): nose → shoulder center → hip center. Subtracts the origin to make coordinates body-relative. Falls back when nose landmark is missing.

**Multi-scale velocity** (`AnatomicalConformer`): three temporal scales are fed to the velocity projections. Δ1 is computed in `ASLDataset` as frame differences of raw (pre-normalization) coordinates. Δ2 and Δ5 are computed inside `AnatomicalConformer.forward()` from body-relative positions (after `RobustNormalization` subtracts the nose), so they capture velocity relative to body movement. All three scales are concatenated per body part before projection.

**Hand dominance** (`HandDominanceModule`): detects dominant hand from wrist velocity; reorders left/right channels so dominant hand is always in the first (LH) slot before projection. Implemented in both `AnatomicalConformer` and the VQ-VAE pipeline. Swap triggers when right-hand energy exceeds left-hand energy — left-handed signers naturally already have dominant hand in the LH slot.

**Soft diversity loss** (`EMAVectorQuantizer`): computed from the distance matrix using `softmax(-distances)` before the argmin step. Gradients flow through `z_flat` to the encoder, pushing it toward spread-out representations. The codebook (EMA buffer) is detached — only the encoder receives this gradient.

**Variable-length batching**: all datasets return a `padding_mask` `(B, T)` bool tensor (`True` = valid). Pass to model alongside `landmarks` or token indices.

**AMP training**: `torch.amp.autocast` + `GradScaler` wrap every forward/backward pass. Disabled automatically when not on CUDA.

**Preprocessing cache**: `VQVAEDataset` accepts `cache_dir`. First access processes each parquet and saves a `.pt` tensor; subsequent accesses skip parquet parsing entirely. Default: `data/cache/`.

**Pre-tokenization cache**: `precompute_tokens.py` runs the frozen VQ-VAE once, saves per-sample token indices to `data/tokens/`. `TokenizedTranslationDataset` loads these directly — no VQ-VAE needed during Phase 2 training.

**Conformer kernel size**: must be significantly smaller than the average sequence length. VQ-VAE chunk size 8 → 40–80 frame signs produce 5–10 tokens. `encoder_kernel_size=7` fits within the sequence; larger values operate mostly on padding.

**Best model selection**: Phase 1 saves checkpoints based on val reconstruction loss (not total loss). The diversity term dominates total loss magnitude and is a poor ranking signal.

**Per-sample augmentation** (`cnn_transformer/train.py`): each sample in a batch gets an independent augmentation decision. Vectorized for flip/noise/rotation; per-sample loop for time_stretch/finger_dropout.

**Label mapping**: 250 ASL signs indexed 0–249. Mapping lives in `data/Isolated_ASL_Recognition/sign_to_prediction_index_map.json`.
