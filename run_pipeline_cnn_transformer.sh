#!/bin/bash
# AnatomicalConformer training pipeline (two-phase)
# Model code lives in research/models/cnn_transformer/
# Runs locally, on RunPod, or any machine with GPU + the dataset.
#
# Usage:
#   bash run_pipeline_cnn_transformer.sh
#   bash run_pipeline_cnn_transformer.sh --phase1-epochs 10 --phase2-epochs 5  # quick test
#   bash run_pipeline_cnn_transformer.sh --data-dir /mnt/data/asl-signs         # custom data path
#   bash run_pipeline_cnn_transformer.sh --skip-lmdb                            # skip LMDB build
#   bash run_pipeline_cnn_transformer.sh --map-size-gb 200                      # larger LMDB on big machines
#   bash run_pipeline_cnn_transformer.sh --allow-errors                         # ignore bad parquets

set -e  # stop on any error

# Make research/models importable as a package root
export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"

# Reduces CUDA memory fragmentation on long runs
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="data/Isolated_ASL_Recognition"
CACHE_DIR="data/cache/cnn_transformer"
LMDB_PATH="data/cache/cnn_transformer/asl.lmdb"
CHECKPOINT_DIR="checkpoints/cnn_transformer"
PHASE1_EPOCHS=100
PHASE2_EPOCHS=20
PATIENCE=20
BATCH_SIZE=64
NUM_WORKERS=4
SKIP_LMDB=false
MAP_SIZE_GB=100
ALLOW_ERRORS=false
LMDB_WORKERS=4

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --cache-dir)      CACHE_DIR="$2";      shift 2 ;;
        --lmdb-path)      LMDB_PATH="$2";      shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --phase1-epochs)  PHASE1_EPOCHS="$2";  shift 2 ;;
        --phase2-epochs)  PHASE2_EPOCHS="$2";  shift 2 ;;
        --patience)       PATIENCE="$2";       shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";     shift 2 ;;
        --num-workers)    NUM_WORKERS="$2";    shift 2 ;;
        --skip-lmdb)      SKIP_LMDB=true;      shift ;;
        --map-size-gb)    MAP_SIZE_GB="$2";    shift 2 ;;
        --allow-errors)   ALLOW_ERRORS=true;        shift ;;
        --lmdb-workers)   LMDB_WORKERS="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$CACHE_DIR" "$CHECKPOINT_DIR"

echo "========================================"
echo "AnatomicalConformer Pipeline"
echo "  Data dir:        $DATA_DIR"
echo "  Cache dir:       $CACHE_DIR"
echo "  LMDB path:       $LMDB_PATH"
echo "  Checkpoint dir:  $CHECKPOINT_DIR"
echo "  Phase 1 epochs:  $PHASE1_EPOCHS"
echo "  Phase 2 epochs:  $PHASE2_EPOCHS"
echo "  Patience:        $PATIENCE"
echo "  Batch size:      $BATCH_SIZE"
echo "  Num workers:     $NUM_WORKERS"
echo "  LMDB map size:   ${MAP_SIZE_GB} GB"
echo "  Allow errors:    $ALLOW_ERRORS"
echo "========================================"

# ── LMDB build (resumable — always invoked so partial builds get completed) ──
# Note: CACHE_VERSION is an MD5 of ALL_COLUMNS. Any config change that affects
# ALL_COLUMNS (INCLUDE_DEPTH, INCLUDE_FACE, FACE_LANDMARK_INDICES) invalidates
# old keys automatically, but stale data still occupies disk. After such a change,
# delete the archive first: rm -rf "$LMDB_PATH"
if [ "$SKIP_LMDB" = false ]; then
    echo ""
    echo "[LMDB] Building / resuming LMDB archive (existing keys are skipped)..."
    uv run python -m cnn_transformer.data.build_lmdb \
        --data-dir "$DATA_DIR" \
        --lmdb-path "$LMDB_PATH" \
        --map-size-gb "$MAP_SIZE_GB" \
        $( [ -n "$LMDB_WORKERS" ] && echo "--num-workers $LMDB_WORKERS" ) \
        $( [ "$ALLOW_ERRORS" = true ] && echo "--allow-errors" )
else
    echo ""
    if [ -n "$LMDB_PATH" ]; then
        echo "[LMDB] Skipped (--skip-lmdb). Using pre-built LMDB at $LMDB_PATH."
    else
        echo "[LMDB] Skipped (--skip-lmdb). Training will fall back to .pt cache."
    fi
fi

# ── Train ────────────────────────────────────────────────────────────────────
echo ""
echo "[Train] Running two-phase AnatomicalConformer training..."
uv run python -m cnn_transformer.train \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    ${LMDB_PATH:+--lmdb-path "$LMDB_PATH"} \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --phase1-epochs "$PHASE1_EPOCHS" \
    --phase2-epochs "$PHASE2_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"

echo ""
echo "========================================"
echo "Training complete. Best model: $CHECKPOINT_DIR/best_final.pth"
echo "========================================"
