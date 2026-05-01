#!/bin/bash
# Full VQ-VAE pipeline: Phase 1 (VQ-VAE) → Pre-tokenize → Phase 2 (Translator)
# Model code lives in research/models/vqvae_seq2seq/
#
# Usage:
#   bash run_pipeline_vqvae_seq2seq.sh
#   bash run_pipeline_vqvae_seq2seq.sh --vqvae-epochs 10 --translator-epochs 10  # quick test
#   bash run_pipeline_vqvae_seq2seq.sh --skip-phase1  # if VQ-VAE checkpoint already exists

set -e  # stop on any error

# Make research/models importable as a package root
export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"

# Reduces CUDA memory fragmentation on long runs
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VQVAE_EPOCHS=100
TRANSLATOR_EPOCHS=100
BATCH_SIZE=32
DATA_DIR="data/Isolated_ASL_Recognition"
CACHE_DIR="data/cache/vqvae"
TOKEN_DIR="data/tokens"
CHECKPOINT_DIR="checkpoints/vqvae"
VQVAE_CHECKPOINT="$CHECKPOINT_DIR/best_model.pt"
SKIP_PHASE1=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --vqvae-epochs)      VQVAE_EPOCHS="$2";      shift 2 ;;
        --translator-epochs) TRANSLATOR_EPOCHS="$2";  shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";         shift 2 ;;
        --checkpoint-dir)    CHECKPOINT_DIR="$2"; VQVAE_CHECKPOINT="$2/best_model.pt"; shift 2 ;;
        --skip-phase1)       SKIP_PHASE1=true;        shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$CACHE_DIR" "$TOKEN_DIR" "$CHECKPOINT_DIR"

echo "========================================"
echo "VQ-VAE Pipeline"
echo "  VQ-VAE epochs:     $VQVAE_EPOCHS"
echo "  Translator epochs: $TRANSLATOR_EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "  Data dir:          $DATA_DIR"
echo "  Cache dir:         $CACHE_DIR"
echo "  Token dir:         $TOKEN_DIR"
echo "  Checkpoint dir:    $CHECKPOINT_DIR"
echo "========================================"

# ── Phase 1: Train VQ-VAE ────────────────────────────────────────────────────
if [ "$SKIP_PHASE1" = false ]; then
    echo ""
    echo "[Phase 1] Training VQ-VAE for $VQVAE_EPOCHS epochs..."
    uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
        --data-dir "$DATA_DIR" \
        --cache-dir "$CACHE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$VQVAE_EPOCHS"
else
    echo ""
    echo "[Phase 1] Skipped (--skip-phase1). Using checkpoint: $VQVAE_CHECKPOINT"
fi

if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE checkpoint not found at $VQVAE_CHECKPOINT"
    exit 1
fi

# ── Pre-tokenize ─────────────────────────────────────────────────────────────
echo ""
echo "[Pre-tokenize] Running VQ-VAE over dataset (skips already-tokenized samples)..."
uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --data-dir "$DATA_DIR" \
    --token-dir "$TOKEN_DIR" \
    --cache-dir "$CACHE_DIR" \
    --num-workers 4

# ── Phase 2: Train Translator ─────────────────────────────────────────────────
echo ""
echo "[Phase 2] Training Translator for $TRANSLATOR_EPOCHS epochs..."
uv run python -m vqvae_seq2seq.translation.train_translator \
    --token-dir "$TOKEN_DIR" \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$TRANSLATOR_EPOCHS"

echo ""
echo "========================================"
echo "Pipeline complete."
echo "========================================"
