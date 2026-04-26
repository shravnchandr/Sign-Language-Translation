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
| Pose | 1024 | Body/hand configuration |
| Motion | 512 | Velocity between frames |
| Dynamics | 256 | Acceleration patterns |
| Face | 256 | Non-manual markers (NMMs) |

Training is fully unsupervised — uses all available datasets including fingerspelling.

### Phase 2 — Sign Translator (`vqvae_seq2seq/translation/`)

Classifies token sequences from the frozen VQ-VAE into 250 ASL signs.

```
Token tuples → FactorizedTokenEmbedding
  → Conformer encoder (CNN + multi-head self-attention)
  → HybridDecoder (CTC + Attention)
  → 250-class output
```

Inference uses beam search with CTC prefix scoring.

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
└── data/
    ├── dataset.py              # VQVAEDataset, TranslationDataset
    ├── preprocessing.py        # RobustPreprocessor, LandmarkProcessor
    ├── signer_split.py         # Train/val/test split by signer
    └── vocabulary.py           # Token-to-gloss mapping

training/                       # Standalone scripts (self-contained for Kaggle)
├── train_vqvae.py
└── train_translator.py

data/
├── Isolated_ASL_Recognition/   # Google ASL Signs — 94k samples, 250 signs (labeled)
├── ASL_Fingerspelling_Recognition/ # Fingerspelling (unlabeled, Phase 1 only)
└── WLASL_Landmarks/            # WLASL preprocessed landmarks

notebooks/                      # Experimentation and visualization
research/                       # Legacy models, prototypes, feature engineering
docs/                           # Pipeline docs, competition notes
```

## Setup

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
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

### Phase 1: VQ-VAE Tokenizer

```bash
uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir ./data/Isolated_ASL_Recognition \
  --cache-dir ./data/cache \
  --epochs 100
```

The first epoch builds a tensor cache under `data/cache/` by processing all parquet files. Subsequent epochs load directly from cache, skipping parquet parsing entirely.

### Phase 2: Sign Translator

```bash
uv run python -m vqvae_seq2seq.translation.train_translator \
  --vqvae-checkpoint ./checkpoints/vqvae/best_model.pt
```

### Standalone scripts (Kaggle / no package install)

```bash
uv run python training/train_vqvae.py --data-dir data/Isolated_ASL_Recognition
uv run python training/train_translator.py \
  --data-dir data/Isolated_ASL_Recognition \
  --vqvae-checkpoint checkpoints/best_model.pt
```

## Key Design Decisions

**Robust normalization** (`RobustPreprocessor`): coordinates are made body-relative using a fallback chain — nose → shoulder center → hip center — to handle partially-detected skeletons.

**Hand dominance** (`HandDominanceModule`): detects the dominant hand from wrist velocity and always places it in the first channel, making the model hand-agnostic by construction.

**Multi-scale temporal encoding**: the same sequence is encoded at chunk sizes 4, 8, and 16 to capture fine finger motion, coarse arm motion, and global temporal structure simultaneously.

**Factorized quantization**: separating pose, motion, dynamics, and face into independent codebooks produces more interpretable tokens and reduces codebook collapse compared to a single shared codebook.

**Variable-length batching**: all datasets return a `padding_mask` tensor `(B, T)` where `True` marks padding positions. Pass alongside `landmarks` to every model forward call.

**AMP training**: `torch.amp.autocast` + `GradScaler` wrap every forward/backward pass. Automatically disabled when not on CUDA, so local CPU/MPS runs are unaffected.

**Preprocessing cache**: `VQVAEDataset` and `TranslationDataset` accept a `cache_dir` argument. On first access each sample is processed from parquet and saved as a `.pt` file; subsequent accesses load the cached tensor directly. Pass `--cache-dir ./data/cache` to the training scripts.

## Code Formatting

```bash
uv run black <file_or_directory>
```
