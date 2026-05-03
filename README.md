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

End-to-end supervised classification with optional CTC pre-training on ASL Fingerspelling. Designed for Kaggle/RunPod training.

```
Parquet → build_lmdb.py (normalize_values: nose→shoulder→hip fallback)
  → LMDB (body-relative coordinates, pre-normalized at build time)
  → ASLDataset: Δ1 velocity as frame-delta of raw coordinates
  → AnatomicalConformer.forward():
      → HandDominanceModule (reorder by wrist motion energy)
      → WristNormalization (lm0=location, lm1-20=shape)
      → Δ2, Δ5 velocity from body-relative positions
      → Per-part position projections: lh, rh, pose, eyebrow, mouth → d_model
      → Per-part velocity projections: lh, rh, pose, face (Δ1∥Δ2∥Δ5) → d_model
      → Geometry stream: 25 features/hand (15 joint-angle cosines + 10 fingertip distances)
      → Feature fusion pos + vel + geo → d_model
      → Sinusoidal PE + CLS token prepend
      → Conformer blocks (FFN→Attention→DepthwiseConv→FFN) × n_layers
      → CLS token → sign head (250 classes)
                 └→ GRL → SignerDiscriminator (n_signers classes, adversarial)
```

**Key regularisation:**
- Stochastic depth: linearly increasing block-skip probability (0 → `drop_path_max=0.1`)
- GRL signer-invariance: adversarial discriminator on CLS token forces features to discard signer identity; λ ramped via Ganin schedule; discriminator accuracy logged each epoch alongside chance level (`1/n_signers`)
- Dominance-aware mixup: pairs same-dominant-hand samples so HandDominanceModule receives unambiguous mixed tensors
- FocalLoss: per-class inverse-frequency weights from train.csv, mean-normalised; γ=2.0, label smoothing=0.1

**Pre-training (optional):**
- CTC pre-training on ASL Fingerspelling: backbone learns fine-grained hand-shape representations before fine-tuning
- Early stopping on val CTC loss with `--pretrain-patience` (default 10); CTC loss computed in FP32 to avoid FP16 underflow
- Saves `backbone_best.pth`; loaded at fine-tuning start via `--pretrained-backbone`

**Training:**
- Phase 1: 100 epochs, heavy augmentation (flip, noise, time-stretch, rotation, finger dropout), dominance-aware mixup, gradient accumulation ×4, OneCycleLR
- Phase 2: 20 epochs, cosine warmdown (heavy augmentation maintained), gradient accumulation ×4; GRL continues Ganin ramp from Phase 1 endpoint (~max_lambda throughout)
- Evaluation: 5-pass test-time augmentation (TTA) for final reporting; deterministic eval for checkpoint selection
- Loss: FocalLoss (per-class inverse-frequency weights, γ=2.0, label smoothing=0.1) + GRL adversarial cross-entropy

**Model size:** ~7M parameters (d_model=256, n_layers=4, n_heads=4)

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
│   │   ├── config.py                   # Landmark layout constants, feature dimensions
│   │   ├── pretrain_fingerspelling.py  # CTC pre-training on ASL Fingerspelling
│   │   ├── train.py                    # Two-phase training loop
│   │   ├── model/
│   │   │   ├── anatomical_conformer.py # Main model
│   │   │   ├── conformer.py            # ConformerBlock, SinusoidalPositionalEncoding
│   │   │   ├── normalization.py        # RobustNormalization, WristNormalization
│   │   │   └── grl.py                  # SignerDiscriminator, ganin_lambda
│   │   └── data/
│   │       ├── dataset.py              # ASLDataset, BucketBatchSampler, get_data_loaders
│   │       ├── augmentation.py         # AdvancedAugmentation (7 types), mixup_batch
│   │       ├── preprocessing.py        # frame_stacked_data
│   │       ├── build_lmdb.py           # One-time LMDB archive builder
│   │       ├── build_fingerspelling_lmdb.py  # Fingerspelling LMDB builder (parquet-file parallelism)
│   │       ├── fingerspelling_dataset.py     # FingerspellingDataset, collate_ctc, load_char_map
│   │       └── _cache_keys.py          # CACHE_VERSION hash, lmdb_key helpers
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
├── asl-is-lmdb/                 # ASL Signs LMDB dataset (recommended — download below)
├── asl-fs-lmdb/                 # Fingerspelling LMDB dataset (recommended — download below)
├── Isolated_ASL_Recognition/    # Raw ASL Signs parquets (only needed to rebuild LMDB)
├── ASL_Fingerspelling_Recognition/ # Raw Fingerspelling parquets (189 GB, rarely needed)
└── WLASL_Landmarks/             # WLASL preprocessed landmarks

run_pipeline_vqvae_seq2seq.sh    # End-to-end VQ-VAE pipeline: Phase 1 → tokenize → Phase 2
run_pipeline_cnn_transformer.sh  # AnatomicalConformer pipeline: LMDB build → two-phase train
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

**Recommended — download pre-built LMDB datasets (no parquet parsing needed):**

```bash
kaggle datasets download shravnchandr/asl-is-lmdb -p data/asl-is-lmdb --unzip
kaggle datasets download shravnchandr/asl-fs-lmdb -p data/asl-fs-lmdb --unzip
```

**Raw competition data (only needed to rebuild LMDBs or run the VQ-VAE pipeline):**

```bash
kaggle competitions download -c asl-signs               # Google ASL Signs (~5 GB)
kaggle competitions download -c asl-fingerspelling       # Fingerspelling (~189 GB)
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
# Recommended: downloaded LMDB datasets, skip pre-training
bash run_pipeline_cnn_transformer.sh --skip-pretrain

# With CTC pre-training using downloaded FS LMDB (no 189 GB download needed)
bash run_pipeline_cnn_transformer.sh

# Re-use an existing backbone checkpoint
bash run_pipeline_cnn_transformer.sh \
  --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth

# Quick smoke test
bash run_pipeline_cnn_transformer.sh --skip-pretrain \
  --phase1-epochs 2 --phase2-epochs 1

# Build LMDBs locally from raw parquets (if you have the competition data)
bash run_pipeline_cnn_transformer.sh --build-lmdb --build-fs-lmdb \
  --data-dir data/Isolated_ASL_Recognition \
  --fs-data-dir data/ASL_Fingerspelling_Recognition
```

**Key pipeline flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--skip-pretrain` | false | Skip FS LMDB build + CTC pre-training |
| `--skip-fs-lmdb` | true | Skip FS LMDB build but still run CTC pre-training |
| `--build-fs-lmdb` | — | Build FS LMDB from raw parquets (overrides `--skip-fs-lmdb`) |
| `--pretrained-backbone <path>` | — | Use existing backbone; implies `--skip-pretrain` |
| `--pretrain-epochs` | 40 | CTC pre-training epochs |
| `--pretrain-patience` | 10 | Early stopping patience on val CTC loss |
| `--fs-data-dir` | `data/asl-fs-lmdb` | Fingerspelling dataset root |
| `--skip-lmdb` | true | Skip ASL LMDB build (use pre-built) |
| `--build-lmdb` | — | Build ASL LMDB from raw parquets (overrides `--skip-lmdb`) |
| `--data-dir` | `data/asl-is-lmdb` | ASL Signs dataset root |
| `--phase1-epochs` | 100 | Fine-tuning Phase 1 epochs |
| `--phase2-epochs` | 20 | Fine-tuning Phase 2 epochs |

**Note:** Any change to `config.py` that affects `ALL_COLUMNS` (face landmark selection, `INCLUDE_DEPTH`, `INCLUDE_FACE`) invalidates the LMDB cache — `CACHE_VERSION` is an MD5 of `ALL_COLUMNS` so stale archives are never silently reused. Delete and rebuild: `rm -rf /tmp/asl.lmdb && python -m cnn_transformer.data.build_lmdb ...`

---

## Key Design Decisions

**Robust normalization**: coordinates are made body-relative using a fallback chain — nose → shoulder center → hip center — to handle partially-detected skeletons.

**Hand dominance** (`HandDominanceModule` in both VQ-VAE and AnatomicalConformer): detects the dominant hand from wrist velocity and always places it in the first channel, making the model hand-agnostic by construction. Swap triggers when right-hand energy exceeds left-hand energy — left-handed signers naturally already have dominant hand in the LH slot.

**Multi-scale velocity**: three temporal scales (Δ1/Δ2/Δ5) are fed to the velocity projections. Δ1 is computed in the dataset as frame differences of raw coordinates. Δ2 and Δ5 are computed inside `AnatomicalConformer.forward()` from body-relative positions (post nose-subtraction), so they capture motion relative to the body. All three scales are concatenated per body part before projection, giving the velocity stream the same `d_model` budget as the position stream.

**Multi-scale temporal encoding**: the same sequence is encoded at chunk sizes 4, 8, and 16 to capture fine finger motion, coarse arm motion, and global temporal structure simultaneously.

**Factorized quantization**: separating pose, motion, dynamics, and face into independent codebooks produces more interpretable tokens and reduces codebook collapse compared to a single shared codebook.

**Soft diversity loss**: `EMAVectorQuantizer` computes a soft entropy loss from the distance matrix (before argmin) that backpropagates to the encoder, pushing it toward more spread-out representations.

**Pre-tokenization**: after Phase 1, `precompute_tokens.py` runs the frozen VQ-VAE once and caches token indices as `.pt` files. Phase 2 training loads these directly — no VQ-VAE in memory, ~5–10× faster per epoch.

**Per-sample augmentation**: each sample in a training batch receives an independent augmentation decision. Vectorized for flip/noise/rotation; per-sample loop for time-stretch and finger dropout.

**Signer-independent splits** (`signer_split.py`): train/val/test are split by signer identity to prevent the model from memorizing individual signing styles.

