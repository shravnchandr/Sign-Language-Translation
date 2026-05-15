# Isolated Sign Language Recognition

Research into signer-independent isolated ASL recognition from MediaPipe landmark sequences. The core problem: models trained on one set of signers generalize poorly to new ones, because they memorize individual signing styles rather than learning sign representations. This project explores three approaches to that generalization problem.

## Results

All runs use a signer-independent val split (no test signer seen during training). Baseline to beat: 1st-place Kaggle solution at **0.8929**.

| Run | Model | Val Accuracy | Key changes |
|-----|-------|-------------|-------------|
| 001 | LandmarkConformer (d_model=512, 49M) | 0.7462 | baseline |
| 002 | LandmarkConformer (d_model=256, 6.5M) | **0.7555** | smaller model, GRL, stochastic depth, dominance-aware mixup |
| 003 | + multi-scale velocity (Δ1/Δ2/Δ5), face split, depth | 0.7432 | slight regression — more features need more epochs to converge |
| 004 | + geometry stream (joint angles + fingertip distances), normalization fix | TBD | next RunPod run |
| — | Factorized VQ-VAE + Conformer | WIP | — |
| — | ST-GCN | — | baseline, not yet run |

## Research Directions

- **Signer invariance** — adversarial GRL discriminator forces the feature extractor to discard signer identity
- **Hand dominance normalization** — always route the dominant hand to the same input channel, making the model left/right-agnostic by construction
- **Discrete motion tokenization** — factorized VQ-VAE separates what the body looks like (pose) from how it moves (motion, dynamics), producing interpretable tokens that may generalize better across signers
- **Multi-scale temporal modeling** — velocity at Δ1/Δ2/Δ5 frames and chunk encoding at 4/8/16 frames capture finger micro-motion, arm movement, and global sign structure simultaneously
- **Geometry-aware features** — joint-angle cosines and fingertip distances give hand shape representations invariant to wrist rotation and signer hand scale

---

## Architectures

### 1 — LandmarkConformer (`research/models/cnn_transformer/`)

End-to-end supervised classification. Processes raw landmarks into three parallel feature streams, fuses them, then runs Conformer blocks over the sequence.

**Three feature streams:**

```
landmarks (T, N, 3)
      │
      ├─ position stream ──────────────────────────────┐
      │   per-part Linear (lh, rh, pose,               │
      │   eyebrow, mouth) → d_model                    │
      │                                                ├─ feat_fuse → d_model → Conformer ×4 → CLS → 250 classes
      ├─ velocity stream (d1 | d2 | d5) ───────────────┤
      │   per-part Linear → d_model                    │
      │                                                │
      └─ geometry stream ──────────────────────────────┘
          joint angles + fingertip distances → d_model/4
```

| Stream | What it captures |
|--------|-----------------|
| Position (d_model) | Body-relative joint locations, split by body part |
| Velocity (d_model) | Multi-scale motion: Δ1 (raw coords), Δ2/Δ5 (body-relative) |
| Geometry (d_model/4) | 15 joint-angle cosines + 10 fingertip distances per hand |

The geometry stream is the key addition: raw XYZ obscures hand shape because it conflates location with configuration. Joint angles and fingertip distances are invariant to wrist rotation and signer hand scale.

**Signer invariance via GRL:** A `SignerDiscriminator` sits on the CLS token with a gradient reversal layer. The feature extractor is penalized for producing representations that distinguish signers — by the end of training, the discriminator performs at chance level, confirming the features have shed signer identity.

**Optional CTC pre-training:** Before supervised fine-tuning, the backbone can be pre-trained on ASL Fingerspelling via CTC loss. This initializes hand-shape representations from a much larger dataset before the model sees the 250-class labels. A warmup phase then freezes the pre-trained backbone while the new classification head trains in, then jointly fine-tunes at a lower backbone learning rate.

**Model:** ~7M parameters — d_model=256, 4 Conformer layers, 4 attention heads.

---

### 2 — Factorized VQ-VAE + Conformer (`research/models/vqvae_seq2seq/`)

Two-phase pipeline: unsupervised discrete tokenization followed by supervised sequence-to-label translation.

**Phase 1 — Factorized VQ-VAE (unsupervised):**

Each chunk of frames is encoded into four independent discrete tokens — one per codebook:

| Codebook | Size | Captures |
|----------|------|---------|
| Pose | 256 | Body/hand configuration |
| Motion | 256 | Velocity between frames |
| Dynamics | 128 | Acceleration patterns |
| Face | 128 | Non-manual markers (expressions, mouthing) |

Separating these into independent codebooks rather than a single shared one reduces codebook collapse and produces more interpretable tokens. The encoder is trained unsupervised, so it can use all available data including fingerspelling.

The VQ-VAE runs once after training to tokenize the entire dataset; Phase 2 loads the cached token indices directly (~5–10× faster per epoch, no VQ-VAE in memory).

**Phase 2 — Conformer Translator (supervised):**

A Conformer encoder (6 layers, d_model=256, kernel=7) reads the 5–10 token sequences, and a hybrid CTC + attention decoder (3 layers) produces the 250-class label. Inference uses beam search with CTC prefix scoring.

---

### 3 — ST-GCN (`research/models/st_gcn/`)

Spatial-Temporal Graph Convolutional Network. Reference baseline.

- 209-node anatomical graph (hands, pose, selected face landmarks)
- 9 ST-GCN blocks in 3 stages: 64 → 128 → 256 channels
- Global average pooling for classification

---

## Key Design Decisions

**Signer-independent splits:** Train/val/test are split by signer identity. Without this, a model can reach high accuracy by memorizing individual signers and will fail badly on new ones.

**Hand dominance normalization:** Wrist velocity determines which hand is dominant; the dominant hand is always placed in the first input channel. Left-handed signers naturally already have their dominant hand in the left-hand slot. This makes the model hand-agnostic without requiring any data augmentation.

**Body-relative coordinates:** Nose position is subtracted at LMDB build time (fallback: shoulder center → hip center → zero) so all coordinates are relative to the signer's body rather than the camera frame.

**Factorized quantization:** A single codebook entangles pose and motion — a sign with similar handshape but different speed would get different tokens. Separate codebooks let the model attend to just the relevant factor.

**Soft diversity loss:** The VQ-VAE encoder receives a gradient from a soft entropy term computed across the distance matrix before the argmin, pushing it toward spread-out representations and away from codebook collapse.

**Dominance-aware mixup:** Mixup pairs samples with the same dominant hand. Mixing a left-dominant sample with a right-dominant one would give `HandDominanceModule` an ambiguous tensor — the paired dominant hand energies would conflict.

---

## Project Structure

```
research/
├── models/
│   ├── cnn_transformer/         # LandmarkConformer (end-to-end)
│   │   ├── config.py                   # Landmark layout constants, feature dimensions
│   │   ├── pretrain_fingerspelling.py  # CTC pre-training on ASL Fingerspelling
│   │   ├── train.py                    # Training loop
│   │   ├── model/
│   │   │   ├── landmark_conformer.py   # Main model
│   │   │   ├── conformer.py            # ConformerBlock, SinusoidalPositionalEncoding
│   │   │   ├── normalization.py        # WristNormalization
│   │   │   └── grl.py                  # SignerDiscriminator, ganin_lambda
│   │   └── data/
│   │       ├── dataset.py              # ASLDataset, BucketBatchSampler, get_data_loaders
│   │       ├── augmentation.py         # AdvancedAugmentation (7 types), mixup_batch
│   │       ├── build_lmdb.py           # One-time LMDB archive builder
│   │       ├── build_fingerspelling_lmdb.py  # Fingerspelling LMDB builder
│   │       └── fingerspelling_dataset.py     # FingerspellingDataset, collate_ctc
│   ├── vqvae_seq2seq/          # Factorized VQ-VAE pipeline
│   │   ├── vqvae/
│   │   │   ├── vqvae_model.py          # Main VQ-VAE model
│   │   │   ├── vector_quantizer.py     # EMAVectorQuantizer, FactorizedVectorQuantizer
│   │   │   ├── multi_scale_encoder.py  # MultiScaleMotionEncoder
│   │   │   ├── face_encoder.py         # FaceNMMEncoder (5 regions)
│   │   │   ├── hand_dominance.py       # HandDominanceModule
│   │   │   └── cross_attention.py      # CrossFactorAttention
│   │   ├── translation/
│   │   │   ├── translator_model.py     # SignTranslator
│   │   │   ├── conformer.py            # Conformer encoder blocks
│   │   │   ├── decoder.py              # HybridDecoder (CTC + attention)
│   │   │   └── beam_search.py          # Beam search with CTC prefix scoring
│   │   └── scripts/
│   │       └── precompute_tokens.py    # Pre-tokenize dataset with frozen VQ-VAE
│   └── st_gcn/                  # ST-GCN baseline
│       ├── graph_structure.py
│       ├── st_gcn_layers.py
│       └── st_gcn_model.py

data/
├── asl-is-lmdb/                 # ASL Signs LMDB (download below)
├── asl-fs-lmdb/                 # Fingerspelling LMDB (download below)
├── Isolated_ASL_Recognition/    # Raw ASL Signs parquets (only needed to rebuild LMDB)
└── ASL_Fingerspelling_Recognition/ # Raw Fingerspelling parquets (189 GB, rarely needed)

run_pipeline_cnn_transformer.sh  # LandmarkConformer: LMDB build → pre-train → fine-tune
run_pipeline_vqvae_seq2seq.sh    # VQ-VAE pipeline: Phase 1 → tokenize → Phase 2
```

---

## Setup

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

## Data

**Recommended — download pre-built LMDB datasets:**

```bash
kaggle datasets download shravnchandr/asl-is-lmdb -p data/asl-is-lmdb --unzip
kaggle datasets download shravnchandr/asl-fs-lmdb -p data/asl-fs-lmdb --unzip
```

**Raw competition data** (only needed to rebuild LMDBs or run the VQ-VAE pipeline):

```bash
kaggle competitions download -c asl-signs               # Google ASL Signs (~5 GB)
kaggle competitions download -c asl-fingerspelling       # Fingerspelling (~189 GB)
```

Extract to `data/Isolated_ASL_Recognition/` and `data/ASL_Fingerspelling_Recognition/` respectively.

**Parquet format:** columns `frame`, `type`, `landmark_index`, `x`, `y`, `z`. Types: `pose` (33), `left_hand` (21), `right_hand` (21), `face` (468) — 543 landmarks per frame.

---

## Training

### LandmarkConformer

```bash
# Recommended: pre-built LMDBs, skip pre-training
bash run_pipeline_cnn_transformer.sh --skip-pretrain

# With CTC pre-training (uses downloaded FS LMDB — no 189 GB download needed)
bash run_pipeline_cnn_transformer.sh

# Re-use an existing backbone checkpoint
bash run_pipeline_cnn_transformer.sh \
  --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth

# Quick smoke test
bash run_pipeline_cnn_transformer.sh --skip-pretrain \
  --phase1-epochs 2 --phase2-epochs 1
```

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--skip-pretrain` | false | Skip CTC pre-training entirely |
| `--pretrained-backbone <path>` | — | Use existing backbone checkpoint |
| `--pretrain-epochs` | 40 | CTC pre-training epochs |
| `--pretrain-patience` | 10 | Early stopping patience (val CTC loss) |
| `--phase1-epochs` | 100 | Fine-tuning epochs |
| `--build-lmdb` | — | Rebuild ASL LMDB from raw parquets |
| `--build-fs-lmdb` | — | Rebuild Fingerspelling LMDB from raw parquets |
| `--data-dir` | `data/asl-is-lmdb` | ASL Signs dataset root |
| `--fs-data-dir` | `data/asl-fs-lmdb` | Fingerspelling dataset root |

> **LMDB cache invalidation:** any change to `config.py` that affects `ALL_COLUMNS` (face landmark selection, `INCLUDE_DEPTH`, `INCLUDE_FACE`) requires rebuilding the LMDB. `CACHE_VERSION` is an MD5 of `ALL_COLUMNS` so stale archives are never silently reused.

### VQ-VAE Pipeline

```bash
# Full pipeline
PYTHONPATH=research/models bash run_pipeline_vqvae_seq2seq.sh

# Phase 1 — train VQ-VAE
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition --cache-dir data/cache --epochs 100

# Tokenize dataset with trained VQ-VAE (run once after Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition --token-dir data/tokens \
  --cache-dir data/cache --num-workers 4

# Phase 2 — train translator
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens --data-dir data/Isolated_ASL_Recognition --epochs 100
```
