# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Isolated Sign Language Recognition system classifying 250 ASL signs from MediaPipe landmarks.

Two approaches are under active development, both under `research/models/`:

| Approach | Location | Status |
|---|---|---|
| Factorized VQ-VAE → Conformer translator | `research/models/vqvae_seq2seq/` | Primary pipeline |
| LandmarkConformer (end-to-end) | `research/models/cnn_transformer/` | Kaggle training target |
| ST-GCN (baseline) | `research/models/st_gcn/` | Experimental |

## Commands

```bash
# Install dependencies (UV package manager, Python 3.14)
uv sync

# LandmarkConformer on a fresh RunPod pod: env + tmux + LMDB download/verify, then train
bash setup_runpod.sh                     # default: saved output of Kaggle notebook shravnchandr/build-islt-lmdb
bash run_pipeline_cnn_transformer.sh --skip-pretrain --num-workers 8   # relaunches in tmux session "islr"
tmux attach -t islr                      # logs also in logs/cnn_transformer_<timestamp>.log; --no-tmux = foreground

# VQ-VAE full pipeline (run from project root — requires research/models/ on PYTHONPATH)
PYTHONPATH=research/models bash run_pipeline_vqvae_seq2seq.sh
PYTHONPATH=research/models bash run_pipeline_vqvae_seq2seq.sh --vqvae-epochs 10 --translator-epochs 10

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

### Approach 2 — LandmarkConformer (`research/models/cnn_transformer/`)

End-to-end supervised classification. Optional CTC pre-training on ASL Fingerspelling initialises the backbone before fine-tuning on the 250-class task. Designed for Kaggle training.

- Input coordinates: x, y, z per landmark (`INCLUDE_DEPTH=True`)
- Per-body-part projection: separate `nn.Linear` for LH, RH, pose; face is split into `eyebrow_proj` (grammatical: questions/negation) and `mouth_proj` (phonological: mouthing), each at `d_model//8` — same total budget as a single face projection
- Multi-scale velocity stream (Δ1/Δ2/Δ5): Δ1 computed in dataset from the stored (body-relative) coords; Δ2 and Δ5 computed inside `forward()` from body-relative positions (after nose subtraction). All three scales concatenated per part and projected to `d_model//4`. Position and velocity get equal `d_model` budget before fusion.
- Geometry stream: per hand, 15 joint-angle cosines (3 per finger, at MCP/PIP/DIP joints) + 10 fingertip pairwise distances + 3 palm-normal components (3D only) = 28 features/hand. Computed in `_hand_geometry()` from wrist-relative fingers after `WristNormalization`. Projected 2 × `d_model//8` = `d_model//4`. Invariant to wrist rotation and signer hand scale — encodes fine-grained hand shape that raw XYZ obscures.
- Distance stream: dominant/non-dominant wrist-to-nose distance + `dom_ratio` (dominant share of wrist energy, 0.5–1) + per-frame hand presence flags (lh, rh) → `dist_proj` → `d_model//8`. Tells the model when hands are in the face region, and distinguishes a missing hand (stored as 0 = nose origin) from a hand at the face.
- Input layout: `(B, T, IN_FEAT)` = `[pos (COORD_FEAT) | Δ1 vel (COORD_FEAT) | presence (2)]` (`config.PRESENCE_START`, `config.IN_FEAT`). Presence is derived in the datasets by `hand_presence()` (see Key Patterns) — no LMDB rebuild needed.
- Feature fusion: pos (`d_model`) + vel (`d_model`) + geo (`d_model//4`) + dist (`d_model//8`) → `feat_fuse` → `d_model`
- Shoulder-width scaling in `forward()`: pos and vel divided by the per-sequence mean shoulder width (camera-distance / body-size invariance).
- Conformer blocks (depthwise conv + self-attention) + CLS token for classification
- Body-relative normalization done once at LMDB build time (`normalize_values`: nose → shoulder → hip → 0 fallback). `WristNormalization` applied in-model: landmark 0 = location (nose-relative), landmarks 1–20 = shape (wrist-relative).
- **Optional fingerspelling CTC pre-training** (`pretrain_fingerspelling.py`): trains the backbone on `ASL_Fingerspelling_Recognition` using `nn.CTCLoss` (60-token char vocab + blank). CTC mode skips the CLS token and returns per-frame logits `(B, T, vocab+1)`. Signer-independent split via `GroupShuffleSplit` on `participant_id`. Early stopping on val CTC loss with `--patience` (default 10). CTC loss computed in FP32 (logits cast before log-softmax to avoid FP16 underflow with long sequences). Saves `backbone_best.pth` (all keys except `head.`, `ctc_head.`, `signer_disc.`, `cls_token`). Loaded at fine-tuning start via `--pretrained-backbone` with `strict=False`.
- **Backbone warmup** (`--backbone-warmup-epochs`, default 5): when `--pretrained-backbone` is set, freezes all backbone params for the first N epochs and trains only the params the checkpoint did not provide (the `missing` keys from `load_state_dict` — always `head.`, `cls_token`, `signer_disc.`, plus any layer added after the backbone was saved). After warmup, backbone is unfrozen with `--backbone-lr-ratio` (default 0.1) × head LR via a separate param group. Patience counter resets at warmup end so early stopping counts from the first joint-training epoch. GRL disabled during warmup (backbone is frozen so adversarial gradients cannot update it). Set `--backbone-warmup-epochs 0` to disable.
- Loss: `FocalLoss` with per-class inverse-frequency weights (computed from train.csv, mean-normalised, registered as buffer), γ=2.0, label smoothing=0.1. Replaces the prior scalar `alpha=0.25` which was a 4× global loss scaling rather than class-imbalance handling.
- Training: Phase 1 (100 epochs default, heavy aug, mixup, OneCycleLR, early stopping on deterministic val acc). Optional Phase 2 cosine warmdown (`--phase2-epochs`, default 0 — it never beat Phase 1 in Runs 002/003).
- Test-time augmentation (5-pass TTA) at evaluation
- Stochastic depth (`drop_path_max=0.1`): linearly increasing per-block skip probability (block 0 = 0, last = drop_path_max). Controlled via `--drop-path-max` CLI arg.
- GRL signer-invariance (`--grl-lambda 0.1`): `SignerDiscriminator` on CLS token, gradient reversed so feature extractor is forced to discard signer identity. λ is applied once, inside the gradient reversal (`loss = sign_loss + adv_loss`), so the backbone receives −λ·∇ and the discriminator trains at full rate. Ganin schedule ramps λ from 0 → max over Phase 1; Phase 2 continues the ramp from Phase 1's endpoint (stays at ~max_lambda). Requires `participant_id` in train.csv; auto-disables when not present. Adversarial loss uses same mixup weighting as sign loss. Discriminator accuracy is logged each epoch alongside chance level (`1/n_signers`) to verify the feature extractor is successfully confusing the discriminator.

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

### LandmarkConformer (`research/models/cnn_transformer/`)

| File | Purpose |
|------|---------|
| `config.py` | Landmark layout constants, feature dimensions |
| `model/landmark_conformer.py` | `LandmarkConformer` — main model |
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
| `research/models/vqvae_seq2seq/vqvae/vqvae_model.py` | 348–353 | `decode()` routes codebooks into fusion slots with mismatched names (pose_q→`dominant_hand`, motion_q→`non_dominant_hand`, dynamics_q→`pose`). Deliberate so every codebook gets reconstruction gradient, but the slot names are misleading — rename the fusion slots to the factor names. |
| `research/models/vqvae_seq2seq/vqvae/hand_dominance.py` | 173–193 | `HandMirrorAugmentation` doesn't correctly flip x-coordinates. |
| `research/models/vqvae_seq2seq/translation/train_translator.py` | — | ~~Chunk size hardcoded as `8` when computing encoder lengths.~~ **Fixed: uses `vqvae.config.base_chunk_size`.** |
| `research/models/st_gcn/st_gcn_model.py` | 106–110 | `edge_importance` parameter is allocated but never used in `forward()`. |
| `research/models/st_gcn/st_gcn_training.py` | 219–225 | Double-normalizes adjacency matrix: `LandmarkGraph.get_normalized_adjacency()` already normalizes, then chain-edges are added and it's normalized again. |
| `research/models/cnn_transformer/data/preprocessing.py` | — | ~~BASE_PATH double-prefixing: `frame_stacked_data` prepended `BASE_PATH` to an already-absolute path built by `dataset.py`. Fixed: `pd.read_parquet(file_path)` directly.~~ **Fixed.** |
| `research/models/cnn_transformer/model/landmark_conformer.py` | — | ~~Hand dominance swap inverted: `lh_energy > rh_energy` triggered swap when LH was already dominant.~~ **Fixed: `rh_energy > lh_energy`.** |
| `research/models/cnn_transformer/model/normalization.py` | — | ~~`RobustNormalization` mutated input in-place, corrupting TTA source tensors.~~ **Fixed: clones output before writing.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`random_flip` double-negated velocity x-coords (two loop passes, each covering full tensor).~~ **Fixed: single pass.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`random_flip` swapped only hand blocks; pose limbs and eyebrows/lip corners stayed un-swapped, producing anatomically impossible mirrors.~~ **Removed: redundant once `HandDominanceModule` mirrors every right-dominant input.** |
| `research/models/cnn_transformer/data/preprocessing.py` | — | ~~A never-detected hand was stored as 0 = "at the nose", indistinguishable from a hand at the face; the FS LMDB also wrote 0 for every missing frame.~~ **Fixed: `hand_presence()` flags + gap filling in both datasets.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`augment_sample` spatial shift drew an independent ±0.02 offset per feature column, distorting hand shape (≈ 0.22 mean change in joint-angle cosines vs 0.28 natural spread).~~ **Fixed: one rigid offset per axis.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`time_stretch` cropped stretched batches back to T, cutting the last 10–23% of nearly every sign (buckets make most samples ≈ T).~~ **Fixed: returns the longer batch.** |
| `research/models/cnn_transformer/model/landmark_conformer.py` | — | ~~`HandDominanceModule` swapped LH/RH slots without mirroring, so the dominant slot held right-hand anatomy for some signers and left-hand anatomy for others.~~ **Fixed: full horizontal mirror (`MIRROR_PERM` + x negation).** |
| `research/models/cnn_transformer/train.py` | — | ~~GRL λ applied twice (inside `grad_reverse` and as `grl_lam * adv_loss`): backbone got −λ², discriminator trained at λ×.~~ **Fixed: `loss = sign_loss + adv_loss`.** |
| `research/models/cnn_transformer/train.py` | — | ~~Per-frame σ=0.01 noise on all features in the train loop — as large as the whole Δ1 velocity signal and ≈ 0.19 change in joint-angle cosines.~~ **Removed** (dataset-level σ=3e-3 noise before velocity remains). |
| `research/models/cnn_transformer/train.py` | — | ~~Train accuracy scored against `y_a` only under mixup, roughly halving it (Runs 002/003 logged ~0.42).~~ **Fixed: `lam·acc_a + (1−lam)·acc_b`.** |
| `research/models/cnn_transformer/train.py` | — | ~~Warmup "head" prefixes hardcoded `lh_geo_proj`/`rh_geo_proj`/`feat_fuse` (present in any current backbone) and omitted `dist_proj`.~~ **Fixed: head = keys missing from the checkpoint.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`mixup_batch` discarded the shuffled sample's mask.~~ **Fixed: returns `mask \| mask[index]`.** |
| `research/models/cnn_transformer/train.py` | — | ~~Class weights built from `value_counts()` of present labels only — an absent class shortened the vector and shifted every later weight onto the wrong class.~~ **Fixed: reindexed over `range(NUM_CLASSES)`.** |
| `research/models/cnn_transformer/train.py` | — | ~~OneCycleLR `steps_per_epoch=len(train_loader)` ignored gradient accumulation, making schedule 2–4× slower.~~ **Fixed: `total_steps` computed from actual optimizer step counts per phase.** |
| `research/models/cnn_transformer/train.py` | — | ~~Phase 2 loop `range(epoch_idx+1, total_steps)` ran too many epochs after early stopping.~~ **Fixed: `range(NUM_EPOCHS_PHASE2)`.** |
| `research/models/cnn_transformer/train.py` | — | ~~Validation used 5× stochastic TTA, making checkpoint selection noisy.~~ **Fixed: `evaluate_epoch` is deterministic; TTA reserved for final reporting via `evaluate_epoch_tta`.** |
| `research/models/cnn_transformer/model/conformer.py` | — | ~~Depthwise conv ran over padded positions, leaking zeros into valid boundary frames.~~ **Fixed: conv residual zeroed at padded positions.** |
| `research/models/cnn_transformer/data/augmentation.py` | — | ~~`mixup_batch` paired lh-dominant with rh-dominant samples, producing ambiguous hand slot assignments when `HandDominanceModule` runs inside the model.~~ **Fixed: dominance-aware pairing shuffles within same-dominance groups.** |
| `research/models/cnn_transformer/data/preprocessing.py` + `model/landmark_conformer.py` | — | ~~Double normalization: `normalize_values` zeroed the nose before LMDB, so `RobustNormalization` in the model always fell through to the shoulder-center branch (nose appeared missing).~~ **Fixed: `normalize_values` implements the full nose→shoulder→hip fallback; `RobustNormalization` removed from model.** LMDB must be rebuilt (`_NORM_VERSION` bump auto-invalidates). |
| `research/models/cnn_transformer/train.py` | — | ~~`FocalLoss` used scalar `alpha=0.25` — a 4× global loss scaling, not per-class weighting.~~ **Fixed: inverse-frequency per-class weights from train.csv, mean-normalised, registered as a buffer.** |
| `research/models/cnn_transformer/train.py` | — | ~~A single NaN batch under fp16 AMP poisoned the conv-module BatchNorm running stats: GradScaler protected the weights, but eval (running stats) collapsed to chance permanently while train mode looked fine (Run 004, epoch 55).~~ **Fixed: bf16 autocast on Ampere+; non-finite-loss guard restores BN buffers and skips the backward.** |
| `research/models/cnn_transformer/train.py` | — | ~~TTA held 5 full-batch logit tensors simultaneously (OOM risk on long sequences).~~ **Fixed: running sum accumulation; only one extra tensor in memory at a time.** |
| `research/models/cnn_transformer/model/conformer.py` | — | ~~`SinusoidalPositionalEncoding` raised `IndexError` when `T > max_len=512`.~~ **Fixed: on-the-fly PE generation in `forward()` without mutating the registered buffer (thread-safe).** |
| `research/models/cnn_transformer/config.py` + `model/landmark_conformer.py` | — | ~~`SELECTED_FACE_INDICES` built by iterating `FACE_LANDMARK_INDICES.values()` — a dict key reordering would silently corrupt the eyebrow/mouth slice in the model.~~ **Fixed: explicit key ordering in config.py + runtime assertion in `LandmarkConformer.__init__`.** |
| `research/models/cnn_transformer/data/dataset.py` | — | ~~LMDB opened with `map_size=1<<40` (1 TiB) on flat files over NAS — caused `lmdb.open()` to hang for 45+ minutes at mmap time.~~ **Fixed: flat-file reads use `os.path.getsize() + 256 MB` as map_size.** |
| `research/models/cnn_transformer/pretrain_fingerspelling.py` | — | ~~CTC loss computed inside `autocast` (FP16) — underflows with long sequences (max_frames=384).~~ **Fixed: `logits.float().log_softmax(-1)` before `CTCLoss`.** |
| `research/models/cnn_transformer/pretrain_fingerspelling.py` | — | ~~`scheduler.step()` called unconditionally even when `GradScaler` skipped `optimizer.step()` on NaN/Inf gradients, desynchronising `OneCycleLR`.~~ **Fixed: compare `scaler.get_scale()` before/after to detect skips.** |
| `research/models/cnn_transformer/pretrain_fingerspelling.py` | — | ~~Last 1–3 batches per epoch dropped when `len(train_loader) % accum_steps != 0` — gradients accumulated but never applied.~~ **Fixed: explicit `_optimizer_step()` flush after the training loop if `remainder > 0`.** |

## Data

### Datasets (under `data/`)

**Pre-built LMDB datasets (recommended — download from Kaggle):**
- `data/asl-is-lmdb/` — `is.lmdb.mdb` + `train.csv` + `sign_to_prediction_index_map.json`
- `data/asl-fs-lmdb/` — `fs.lmdb.mdb` + `train.csv` + `character_to_prediction_index.json`

```bash
kaggle datasets download shravnchandr/asl-is-lmdb -p data/asl-is-lmdb --unzip
kaggle datasets download shravnchandr/asl-fs-lmdb -p data/asl-fs-lmdb --unzip
```

**Raw competition data (only needed to rebuild LMDBs or for VQ-VAE pipeline):**
- `data/Isolated_ASL_Recognition/` — Google ASL Signs (94k parquets, 250 signs)
- `data/ASL_Fingerspelling_Recognition/` — Fingerspelling parquets (189 GB)
- `data/WLASL_Landmarks/` — WLASL landmarks after MediaPipe preprocessing

### Parquet Format
Columns: `frame`, `type`, `landmark_index`, `x`, `y`, `z`
- `type`: `'pose'` (33), `'left_hand'` (21), `'right_hand'` (21), `'face'` (478) — 553 total per frame
- Coordinates normalized to [0, 1] by MediaPipe; further normalized body-relative at LMDB build time

## Key Patterns

**Normalization fallback chain** (`RobustPreprocessor`, `RobustNormalization`): nose → shoulder center → hip center. Subtracts the origin to make coordinates body-relative. Falls back when nose landmark is missing.

**Multi-scale velocity** (`LandmarkConformer`): three temporal scales are fed to the velocity projections. Δ1 is computed in `ASLDataset` as frame differences of the stored coordinates (already nose-relative from LMDB build). Δ2 and Δ5 are computed inside `LandmarkConformer.forward()` from body-relative positions after wrist normalization and shoulder-width scaling, so they capture velocity relative to body movement. All three scales are concatenated per body part before projection.

**Hand dominance** (`HandDominanceModule`): detects dominant hand from wrist velocity. In `LandmarkConformer`, sequences where right-hand energy exceeds left-hand energy are horizontally mirrored (negate x, swap every bilateral landmark via `config.MIRROR_PERM`), so the dominant hand always arrives in the first (LH) slot with the same anatomy. There is therefore no mirror-flip augmentation (it would be undone on all but near-tie sequences). The presence flags are swapped with their hands. The VQ-VAE pipeline still uses its own slot-reordering implementation.

**Hand presence** (`data/preprocessing.py:hand_presence`): neither LMDB stores NaNs. The ASL LMDB ffill/bfills gaps (a missing frame is a bit-exact copy of its neighbour); the FS LMDB writes 0 per missing frame. A real detection is never all-zero nor bit-identical to the previous frame, so `present = block ≠ 0 and block ≠ previous block` (a leading identical run is resolved as bfill). Absent frames are then filled by holding the nearest detection so both datasets share one convention. Must run on stored coords **before** `augment_sample` noise. Validated: 2/1064 flag errors on ASL-format parquets, 0/27339 on FS frames.

**Soft diversity loss** (`EMAVectorQuantizer`): computed from the distance matrix using `softmax(-distances)` before the argmin step. Gradients flow through `z_flat` to the encoder, pushing it toward spread-out representations. The codebook (EMA buffer) is detached — only the encoder receives this gradient.

**Variable-length batching**: all datasets return a `padding_mask` `(B, T)` bool tensor (`True` = valid). Pass to model alongside `landmarks` or token indices.

**AMP training**: `torch.amp.autocast` wraps every forward pass — bf16 on Ampere+ (no `GradScaler` needed), fp16 + `GradScaler` otherwise. Disabled automatically when not on CUDA. `train_epoch` skips non-finite-loss batches and restores BatchNorm running stats from a pre-forward snapshot (a NaN forward would otherwise permanently break eval).

**Preprocessing cache**: `VQVAEDataset` accepts `cache_dir`. First access processes each parquet and saves a `.pt` tensor; subsequent accesses skip parquet parsing entirely. Default: `data/cache/`.

**Pre-tokenization cache**: `precompute_tokens.py` runs the frozen VQ-VAE once, saves per-sample token indices to `data/tokens/`. `TokenizedTranslationDataset` loads these directly — no VQ-VAE needed during Phase 2 training.

**Conformer kernel size**: must be significantly smaller than the average sequence length. VQ-VAE chunk size 8 → 40–80 frame signs produce 5–10 tokens. `encoder_kernel_size=7` fits within the sequence; larger values operate mostly on padding.

**Best model selection**: Phase 1 saves checkpoints based on val reconstruction loss (not total loss). The diversity term dominates total loss magnitude and is a poor ranking signal.

**Per-sample augmentation** (`cnn_transformer/train.py`): each sample in a batch gets an independent augmentation decision. Vectorized for rotation/finger dropout; time_stretch is batch-wide. Geometric transforms touch only coordinate channels (`:PRESENCE_START`), never the presence flags. Coordinate noise is added once, in `ASLDataset` (`augment_sample`, before velocity is computed).

**Label mapping**: 250 ASL signs indexed 0–249. Mapping lives in `data/asl-is-lmdb/sign_to_prediction_index_map.json` (downloaded with the LMDB dataset).
