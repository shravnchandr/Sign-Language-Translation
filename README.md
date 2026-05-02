# Isolated Sign Language Recognition

Classifies 250 ASL signs from MediaPipe landmark sequences. Two model architectures are under development: a two-phase Factorized VQ-VAE pipeline and a direct end-to-end AnatomicalConformer.

## Architectures

### Approach 1 — Factorized VQ-VAE Pipeline (`research/models/vqvae_seq2seq/`)

Unsupervised pre-training followed by supervised translation.

```
Parquet → LandmarkProcessor → (T, 543, 3)
  → RobustPreprocessor (normalize, augment)
  → HandDominanceModule (reorder dominant/non-dominant)
  → MultiScaleMotionEncoder (chunk sizes: 4, 8, 16)
  → CrossFactorAttention (fuse pose/motion/dynamics)
  → FactorizedVectorQuantizer
  → [(pose_id, motion_id, dynamics_id, face_id), ...]  [saved to data/tokens/]
  → FactorizedTokenEmbedding → Conformer → HybridDecoder → 250-class output
```

**Factorized codebooks** disentangle what the body looks like from how it moves:

| Codebook | Size | Captures |
|----------|------|---------|
| Pose | 256 | Body/hand configuration |
| Motion | 256 | Velocity between frames |
| Dynamics | 128 | Acceleration patterns |
| Face | 128 | Non-manual markers (NMMs) |

Phase 1 training is fully unsupervised — uses all available datasets including fingerspelling.

**Translator config** (sized for 5–10 token sequences per sign):

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_encoder_layers` | 6 |
| `n_decoder_layers` | 3 |
| `encoder_kernel_size` | 7 |
| Parameters | ~18M |

---

### Approach 2 — AnatomicalConformer (`research/models/cnn_transformer/`)

End-to-end supervised classification without VQ-VAE pre-training. Designed for Kaggle/RunPod training.

```
Parquet → frame_stacked_data → (T, D_pos) positions
  → velocity computed as frame-delta of positions
  → concat [pos | vel] → (T, 2·D_pos)
  → RobustNormalization (nose→shoulder fallback, in-model)
  → HandDominanceModule (reorder by wrist motion energy)
  → WristNormalization (lm0=location, lm1-20=shape)
  → Per-part projections: lh, rh, pose, face (pos + vel streams)
  → Feature fusion (pos+vel → d_model)
  → Sinusoidal PE + CLS token prepend
  → Conformer blocks (FFN→Attention→DepthwiseConv→FFN) × n_layers
  → CLS token → sign head (250 classes)
             └→ GRL → SignerDiscriminator (n_signers classes, adversarial)
```

**Key regularisation:**
- Stochastic depth: linearly increasing block-skip probability (0 → `drop_path_max=0.1`)
- GRL signer-invariance: adversarial discriminator on CLS token forces features to discard signer identity; λ ramped via Ganin schedule
- Dominance-aware mixup: pairs same-dominant-hand samples so HandDominanceModule receives unambiguous mixed tensors

**Training:**
- Phase 1: 80 epochs, heavy augmentation (flip, noise, time-stretch, rotation, finger dropout), dominance-aware mixup, gradient accumulation ×4, OneCycleLR
- Phase 2: 20 epochs, cosine warmdown (heavy augmentation maintained), gradient accumulation ×4
- Evaluation: 5-pass test-time augmentation (TTA) for final reporting; deterministic eval for checkpoint selection
- Loss: FocalLoss (α=0.25, γ=2.0, label smoothing=0.1) + GRL adversarial cross-entropy

**Model size:** ~6.5M parameters (d_model=256, n_layers=4, n_heads=4)

---

### Approach 3 — ST-GCN (`research/models/st_gcn/`)

Spatial-Temporal Graph Convolutional Network. Baseline / reference implementation.

- 209-node anatomical graph (hands, pose, selected face landmarks)
- 9 ST-GCN blocks in 3 stages: 64 → 128 → 256 channels
- Global average pooling for final representation

---

## Project Structure

```
research/
├── models/
│   ├── vqvae_seq2seq/          # Factorized VQ-VAE pipeline
│   │   ├── vqvae/
│   │   │   ├── config.py               # ImprovedVQVAEConfig
│   │   │   ├── vqvae_model.py          # Main VQ-VAE model
│   │   │   ├── vector_quantizer.py     # EMAVectorQuantizer, FactorizedVectorQuantizer
│   │   │   ├── multi_scale_encoder.py  # MultiScaleMotionEncoder
│   │   │   ├── face_encoder.py         # FaceNMMEncoder (5 regions)
│   │   │   ├── hand_dominance.py       # HandDominanceModule
│   │   │   ├── cross_attention.py      # CrossFactorAttention
│   │   │   └── train_vqvae.py          # Training entry point
│   │   ├── translation/
│   │   │   ├── config.py               # TranslationConfig
│   │   │   ├── translator_model.py     # SignTranslator
│   │   │   ├── conformer.py            # Conformer encoder blocks
│   │   │   ├── decoder.py              # HybridDecoder (CTC + Attention)
│   │   │   ├── beam_search.py          # BeamSearch with CTC prefix scoring
│   │   │   ├── token_embedding.py      # FactorizedTokenEmbedding
│   │   │   └── train_translator.py     # Training entry point
│   │   ├── scripts/
│   │   │   └── precompute_tokens.py    # Pre-tokenize dataset with frozen VQ-VAE
│   │   └── data/
│   │       ├── dataset.py              # VQVAEDataset, TranslationDataset, TokenizedTranslationDataset
│   │       ├── preprocessing.py        # RobustPreprocessor, LandmarkProcessor
│   │       ├── signer_split.py         # Signer-independent train/val/test split
│   │       └── vocabulary.py           # Token-to-gloss mapping
│   ├── cnn_transformer/         # AnatomicalConformer (end-to-end)
│   │   ├── config.py                   # Landmark layout constants
│   │   ├── train.py                    # Two-phase training loop
│   │   ├── model/
│   │   │   ├── anatomical_conformer.py # Main model
│   │   │   ├── conformer.py            # ConformerBlock, SinusoidalPositionalEncoding
│   │   │   └── normalization.py        # RobustNormalization
│   │   └── data/
│   │       ├── dataset.py              # ASLDataset, BucketBatchSampler
│   │       ├── augmentation.py         # AdvancedAugmentation (7 types), mixup_batch
│   │       └── preprocessing.py        # frame_stacked_data
│   └── st_gcn/                  # ST-GCN baseline
│       ├── graph_structure.py          # LandmarkGraph (anatomical adjacency)
│       ├── st_gcn_layers.py            # GraphConvolution, TemporalConvolution, ST_GCN_Block
│       ├── st_gcn_model.py             # ST_GCN_ASL, LightweightST_GCN
│       ├── st_gcn_training.py          # Training loop
│       └── data_prep.py                # Data loading adapter
├── feature_engineering/         # Feature extraction experiments
├── prototypes/                  # Early prototype scripts
└── tools/                       # Data visualization and conversion utilities

data/
├── Isolated_ASL_Recognition/    # Google ASL Signs — 94k samples, 250 signs (labeled)
├── ASL_Fingerspelling_Recognition/ # Fingerspelling (unlabeled, Phase 1 only)
└── WLASL_Landmarks/             # WLASL preprocessed landmarks

run_pipeline.sh                  # End-to-end: Phase 1 → pre-tokenize → Phase 2
vqvae_seq2seq_backup/            # Backup of VQ-VAE package
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

```bash
kaggle competitions download -c asl-signs               # Google ASL Signs (labeled)
kaggle competitions download -c asl-fingerspelling       # Fingerspelling (unlabeled pre-training)
```

Extract to `data/Isolated_ASL_Recognition/` and `data/ASL_Fingerspelling_Recognition/` respectively.

**Parquet format:** Each file contains columns `frame`, `type`, `landmark_index`, `x`, `y`, `z`.  
Types: `pose` (33 landmarks), `left_hand` (21), `right_hand` (21), `face` (468) — 543 total per frame.

---

## Training

### VQ-VAE Pipeline (run from project root)

```bash
# Full pipeline
PYTHONPATH=research/models bash run_pipeline.sh

# Quick test
PYTHONPATH=research/models bash run_pipeline.sh --vqvae-epochs 10 --translator-epochs 10

# Phase 1 — VQ-VAE
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition \
  --cache-dir data/cache \
  --epochs 100

# Pre-tokenize (run once after Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition \
  --token-dir data/tokens \
  --cache-dir data/cache \
  --num-workers 4

# Phase 2 — Translator
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens \
  --data-dir data/Isolated_ASL_Recognition \
  --epochs 100
```

### AnatomicalConformer (RunPod / local)

```bash
# Recommended: build LMDB cache once (~2 hrs first run, then instant)
PYTHONPATH=research/models uv run python -m cnn_transformer.data.build_lmdb \
  --data-dir data/Isolated_ASL_Recognition \
  --lmdb-path /tmp/asl.lmdb

# Train (default: d_model=256, n_layers=4, n_heads=4, drop_path_max=0.1, grl_lambda=0.1)
PYTHONPATH=research/models uv run python -m cnn_transformer.train \
  --data-dir data/Isolated_ASL_Recognition \
  --lmdb-path /tmp/asl.lmdb \
  --checkpoint-dir checkpoints/cnn_transformer

# Key CLI args
#   --d-model 256           model width
#   --n-layers 4            conformer block count
#   --drop-path-max 0.1     stochastic depth max rate
#   --grl-lambda 0.1        GRL signer-adversarial weight (0=disable)
#   --phase1-epochs 80
#   --phase2-epochs 20
```

**Note:** Changing face landmark selection in `config.py` (e.g. `FACE_LANDMARK_INDICES`) invalidates the LMDB cache. Delete and rebuild: `rm -rf /tmp/asl.lmdb && python -m cnn_transformer.data.build_lmdb ...`

---

## Key Design Decisions

**Robust normalization**: coordinates are made body-relative using a fallback chain — nose → shoulder center → hip center — to handle partially-detected skeletons.

**Hand dominance** (`HandDominanceModule` in VQ-VAE, planned for AnatomicalConformer): detects the dominant hand from wrist velocity and always places it in the first channel, making the model hand-agnostic by construction.

**Velocity is body-relative**: velocities are computed as frame differences of nose-subtracted positions, giving `world_vel − body_translation`. This means relative hand-to-body motion is already encoded without additional computation.

**Multi-scale temporal encoding**: the same sequence is encoded at chunk sizes 4, 8, and 16 to capture fine finger motion, coarse arm motion, and global temporal structure simultaneously.

**Factorized quantization**: separating pose, motion, dynamics, and face into independent codebooks produces more interpretable tokens and reduces codebook collapse compared to a single shared codebook.

**Soft diversity loss**: `EMAVectorQuantizer` computes a soft entropy loss from the distance matrix (before argmin) that backpropagates to the encoder, pushing it toward more spread-out representations.

**Pre-tokenization**: after Phase 1, `precompute_tokens.py` runs the frozen VQ-VAE once and caches token indices as `.pt` files. Phase 2 training loads these directly — no VQ-VAE in memory, ~5–10× faster per epoch.

**Per-sample augmentation**: each sample in a training batch receives an independent augmentation decision. Vectorized for flip/noise/rotation; per-sample loop for time-stretch and finger dropout.

**Signer-independent splits** (`signer_split.py`): train/val/test are split by signer identity to prevent the model from memorizing individual signing styles.

## Code Formatting

```bash
uv run black <file_or_directory>
```
