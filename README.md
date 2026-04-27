# Isolated Sign Language Recognition

Classifies 250 ASL signs from MediaPipe landmark sequences using a two-phase pipeline: unsupervised Factorized VQ-VAE pre-training followed by a supervised Conformer-based translator.

## Architecture

### Phase 1 — Factorized VQ-VAE (`vqvae_seq2seq/vqvae/`)

Encodes 3D landmark sequences into compact discrete token tuples without labels.

```
Parquet → LandmarkProcessor → (T, 543, 3)
  → RobustPreprocessor (normalize, augment)
  → HandDominanceModule (reorder dominant/non-dominant)
  → MultiScaleMotionEncoder (chunk sizes: 4, 8, 16)
  → CrossFactorAttention (fuse pose/motion/dynamics)
  → FactorizedVectorQuantizer
  → [(pose_id, motion_id, dynamics_id, face_id), ...]
```

**Factorized codebooks** disentangle what the body looks like from how it moves:

| Codebook | Size | Captures |
|----------|------|---------|
| Pose | 256 | Body/hand configuration |
| Motion | 256 | Velocity between frames |
| Dynamics | 128 | Acceleration patterns |
| Face | 128 | Non-manual markers (NMMs) |

Training is fully unsupervised — uses all available datasets including fingerspelling.

### Phase 2 — Sign Translator (`vqvae_seq2seq/translation/`)

Classifies token sequences from the frozen VQ-VAE into 250 ASL signs.

```
Token tuples → FactorizedTokenEmbedding
  → Conformer encoder (CNN + multi-head self-attention, kernel=7)
  → HybridDecoder (CTC + Attention)
  → 250-class output
```

Inference uses beam search with CTC prefix scoring.

**Translator config** (sized for 5–10 token sequences per sign):

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_encoder_layers` | 6 |
| `n_decoder_layers` | 3 |
| `encoder_kernel_size` | 7 |
| Parameters | ~18M |

## Project Structure

```
vqvae_seq2seq/
├── vqvae/
│   ├── config.py               # ImprovedVQVAEConfig — all hyperparameters
│   ├── vqvae_model.py          # Main VQ-VAE model
│   ├── vector_quantizer.py     # EMAVectorQuantizer, FactorizedVectorQuantizer
│   ├── multi_scale_encoder.py  # MultiScaleMotionEncoder
│   ├── face_encoder.py         # FaceNMMEncoder (5 regions)
│   ├── hand_dominance.py       # HandDominanceModule
│   ├── cross_attention.py      # CrossFactorAttention
│   └── train_vqvae.py          # Training entry point
├── translation/
│   ├── config.py               # TranslationConfig
│   ├── translator_model.py     # SignTranslator (full model)
│   ├── conformer.py            # Conformer encoder blocks
│   ├── decoder.py              # HybridDecoder (CTC + Attention)
│   ├── beam_search.py          # BeamSearch with CTC prefix scoring
│   ├── token_embedding.py      # FactorizedTokenEmbedding
│   └── train_translator.py     # Training entry point
├── scripts/
│   └── precompute_tokens.py    # Pre-tokenize dataset with frozen VQ-VAE
└── data/
    ├── dataset.py              # VQVAEDataset, TranslationDataset, TokenizedTranslationDataset
    ├── preprocessing.py        # RobustPreprocessor, LandmarkProcessor
    ├── signer_split.py         # Train/val/test split by signer
    └── vocabulary.py           # Token-to-gloss mapping

training/                       # Standalone scripts (self-contained for Kaggle)
├── train_vqvae.py
└── train_translator.py

run_pipeline.sh                 # End-to-end: Phase 1 → pre-tokenize → Phase 2

data/
├── Isolated_ASL_Recognition/   # Google ASL Signs — 94k samples, 250 signs (labeled)
├── ASL_Fingerspelling_Recognition/ # Fingerspelling (unlabeled, Phase 1 only)
└── WLASL_Landmarks/            # WLASL preprocessed landmarks
```

## Setup

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

## Data

Download the datasets via the Kaggle API:

```bash
kaggle competitions download -c asl-signs               # Google ASL Signs (labeled)
kaggle competitions download -c asl-fingerspelling       # Fingerspelling (unlabeled pre-training)
```

Extract to `data/Isolated_ASL_Recognition/` and `data/ASL_Fingerspelling_Recognition/` respectively.

**Parquet format:** Each file contains columns `frame`, `type`, `landmark_index`, `x`, `y`, `z`.
Types: `pose` (33 landmarks), `left_hand` (21), `right_hand` (21), `face` (468) — 543 total per frame.

## Training

### Full pipeline (recommended)

```bash
# Runs Phase 1 → pre-tokenize → Phase 2 sequentially, stops on any error
bash run_pipeline.sh

# Quick test run
bash run_pipeline.sh --vqvae-epochs 10 --translator-epochs 10

# Custom epochs
bash run_pipeline.sh --vqvae-epochs 100 --translator-epochs 100 --batch-size 64
```

### Step by step

**Phase 1 — VQ-VAE:**
```bash
uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition \
  --cache-dir data/cache \
  --epochs 100
```

The first epoch builds a tensor cache under `data/cache/`. Subsequent epochs load directly from cache.

**Pre-tokenize** (run once after Phase 1):
```bash
uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition \
  --token-dir data/tokens \
  --cache-dir data/cache \
  --num-workers 4
```

Saves per-sample token indices under `data/tokens/`. Idempotent — skips already-tokenized samples.

**Phase 2 — Translator:**
```bash
uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens \
  --data-dir data/Isolated_ASL_Recognition \
  --epochs 100
```

## Key Design Decisions

**Robust normalization** (`RobustPreprocessor`): coordinates are made body-relative using a fallback chain — nose → shoulder center → hip center — to handle partially-detected skeletons.

**Hand dominance** (`HandDominanceModule`): detects the dominant hand from wrist velocity and always places it in the first channel, making the model hand-agnostic by construction.

**Multi-scale temporal encoding**: the same sequence is encoded at chunk sizes 4, 8, and 16 to capture fine finger motion, coarse arm motion, and global temporal structure simultaneously.

**Factorized quantization**: separating pose, motion, dynamics, and face into independent codebooks produces more interpretable tokens and reduces codebook collapse compared to a single shared codebook.

**Soft diversity loss**: `EMAVectorQuantizer` computes a soft entropy loss from the distance matrix (before argmin) that backpropagates to the encoder, pushing it toward more spread-out representations and reducing codebook collapse.

**Pre-tokenization**: after Phase 1, `precompute_tokens.py` runs the frozen VQ-VAE once over the full dataset and caches token indices as `.pt` files. Phase 2 training loads these directly — no VQ-VAE in memory, ~5–10× faster per epoch.

**Variable-length batching**: all datasets return a `padding_mask` tensor `(B, T)` where `True` marks valid positions. Pass alongside inputs to every model forward call.

**AMP training**: `torch.amp.autocast` + `GradScaler` wrap every forward/backward pass. Automatically disabled when not on CUDA.

**Preprocessing cache**: `VQVAEDataset` accepts a `cache_dir` argument. On first access each parquet file is processed and saved as a `.pt` tensor; subsequent accesses skip parsing entirely.

**Conformer kernel sizing**: `encoder_kernel_size=7` is chosen to fit within the 5–10 token sequences produced by the VQ-VAE (40–80 frame signs ÷ chunk size 8). Larger kernels operate mostly on padding and degrade local temporal modeling.

## Code Formatting

```bash
uv run black <file_or_directory>
```
