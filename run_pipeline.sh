#!/bin/bash
# Full pipeline: Phase 1 (VQ-VAE) → Pre-tokenize → Phase 2 (Translator)
# Usage: bash run_pipeline.sh [--vqvae-epochs N] [--translator-epochs N] [--batch-size N]

set -e  # stop on any error

# Reduces CUDA memory fragmentation on long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VQVAE_EPOCHS=100
TRANSLATOR_EPOCHS=100
BATCH_SIZE=64
DATA_DIR="data/Isolated_ASL_Recognition"
CACHE_DIR="data/cache"
TOKEN_DIR="data/tokens"
VQVAE_CHECKPOINT="checkpoints/vqvae/best_model.pt"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --vqvae-epochs)     VQVAE_EPOCHS="$2";     shift 2 ;;
        --translator-epochs) TRANSLATOR_EPOCHS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Pipeline config:"
echo "  VQ-VAE epochs:     $VQVAE_EPOCHS"
echo "  Translator epochs: $TRANSLATOR_EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "========================================"

# ── Phase 1: Train VQ-VAE ────────────────────────────────────────────────────
echo ""
echo "[Phase 1] Training VQ-VAE for $VQVAE_EPOCHS epochs..."
uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$VQVAE_EPOCHS"

# ── Pre-tokenize ─────────────────────────────────────────────────────────────
echo ""
echo "[Pre-tokenize] Running VQ-VAE over dataset..."
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
