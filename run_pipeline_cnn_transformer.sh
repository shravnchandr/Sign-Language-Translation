#!/bin/bash
# AnatomicalConformer training pipeline
# Stages:
#   0. Fingerspelling LMDB build + CTC pre-training (optional)
#   1. ASL LMDB build
#   2. Fine-tuning (with backbone from stage 0 if available)
#
# Usage:
#   bash run_pipeline_cnn_transformer.sh
#   bash run_pipeline_cnn_transformer.sh --skip-pretrain               # skip FS LMDB + pre-training
#   bash run_pipeline_cnn_transformer.sh --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth
#   bash run_pipeline_cnn_transformer.sh --phase1-epochs 10 --phase2-epochs 5  # quick test
#   bash run_pipeline_cnn_transformer.sh --data-dir /mnt/data/asl-signs
#   bash run_pipeline_cnn_transformer.sh --skip-lmdb                   # skip ASL LMDB build
#   bash run_pipeline_cnn_transformer.sh --map-size-gb 200
#   bash run_pipeline_cnn_transformer.sh --allow-errors

set -e

export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Defaults ────────────────────────────────────────────────────────────────
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
MAP_SIZE_GB=""   # empty = 1 TiB default (sparse file, no disk cost)
ALLOW_ERRORS=false
LMDB_WORKERS=4
COMPILE=false

# Fingerspelling pre-training
FS_DATA_DIR="data/ASL_Fingerspelling_Recognition"
FS_LMDB_PATH="data/cache/fingerspelling/fs.lmdb"
FS_LMDB_CSV="data/cache/fingerspelling/train.csv"
PRETRAIN_CHECKPOINT_DIR="checkpoints/pretrain_fs"
PRETRAIN_EPOCHS=40
PRETRAINED_BACKBONE=""
SKIP_PRETRAIN=false
FS_MAP_SIZE_GB=""  # empty = 1 TiB default

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)                DATA_DIR="$2";                shift 2 ;;
        --cache-dir)               CACHE_DIR="$2";               shift 2 ;;
        --lmdb-path)               LMDB_PATH="$2";               shift 2 ;;
        --checkpoint-dir)          CHECKPOINT_DIR="$2";          shift 2 ;;
        --phase1-epochs)           PHASE1_EPOCHS="$2";           shift 2 ;;
        --phase2-epochs)           PHASE2_EPOCHS="$2";           shift 2 ;;
        --patience)                PATIENCE="$2";                shift 2 ;;
        --batch-size)              BATCH_SIZE="$2";              shift 2 ;;
        --num-workers)             NUM_WORKERS="$2";             shift 2 ;;
        --skip-lmdb)               SKIP_LMDB=true;               shift ;;
        --map-size-gb)             MAP_SIZE_GB="$2";             shift 2 ;;
        --allow-errors)            ALLOW_ERRORS=true;            shift ;;
        --lmdb-workers)            LMDB_WORKERS="$2";            shift 2 ;;
        --fs-data-dir)             FS_DATA_DIR="$2";             shift 2 ;;
        --fs-lmdb-path)            FS_LMDB_PATH="$2";            shift 2 ;;
        --fs-lmdb-csv)             FS_LMDB_CSV="$2";             shift 2 ;;
        --pretrain-checkpoint-dir) PRETRAIN_CHECKPOINT_DIR="$2"; shift 2 ;;
        --pretrain-epochs)         PRETRAIN_EPOCHS="$2";         shift 2 ;;
        --pretrained-backbone)     PRETRAINED_BACKBONE="$2";     shift 2 ;;
        --skip-pretrain)           SKIP_PRETRAIN=true;           shift ;;
        --fs-map-size-gb)          FS_MAP_SIZE_GB="$2";          shift 2 ;;
        --compile)                 COMPILE=true;                  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# If --pretrained-backbone is given directly, skip the pre-training stage
if [ -n "$PRETRAINED_BACKBONE" ]; then
    SKIP_PRETRAIN=true
fi

mkdir -p "$CACHE_DIR" "$CHECKPOINT_DIR"

echo "========================================"
echo "AnatomicalConformer Pipeline"
echo "  Data dir:           $DATA_DIR"
echo "  Cache dir:          $CACHE_DIR"
echo "  LMDB path:          $LMDB_PATH"
echo "  Checkpoint dir:     $CHECKPOINT_DIR"
echo "  Phase 1 epochs:     $PHASE1_EPOCHS"
echo "  Phase 2 epochs:     $PHASE2_EPOCHS"
echo "  Patience:           $PATIENCE"
echo "  Batch size:         $BATCH_SIZE"
echo "  Num workers:        $NUM_WORKERS"
echo "  LMDB map size:      ${MAP_SIZE_GB:-1 TiB (default)}"
echo "  Allow errors:       $ALLOW_ERRORS"
echo "  torch.compile:      $COMPILE"
echo ""
if [ "$SKIP_PRETRAIN" = false ]; then
    echo "  [Pre-training]"
    echo "  FS data dir:        $FS_DATA_DIR"
    echo "  FS LMDB path:       $FS_LMDB_PATH"
    echo "  FS LMDB CSV:        $FS_LMDB_CSV"
    echo "  Pretrain ckpt dir:  $PRETRAIN_CHECKPOINT_DIR"
    echo "  Pretrain epochs:    $PRETRAIN_EPOCHS"
    echo "  FS LMDB map size:   ${FS_MAP_SIZE_GB:-1 TiB (default)}"
elif [ -n "$PRETRAINED_BACKBONE" ]; then
    echo "  Pretrained backbone: $PRETRAINED_BACKBONE"
else
    echo "  Pre-training:       skipped (no backbone)"
fi
echo "========================================"

# ── Stage 0a: Fingerspelling LMDB ────────────────────────────────────────────
if [ "$SKIP_PRETRAIN" = false ]; then
    echo ""
    echo "[FS LMDB] Building / resuming fingerspelling LMDB..."
    mkdir -p "$(dirname "$FS_LMDB_PATH")" "$(dirname "$FS_LMDB_CSV")"
    uv run python -m cnn_transformer.data.build_fingerspelling_lmdb \
        --data-dir  "$FS_DATA_DIR" \
        --lmdb-path "$FS_LMDB_PATH" \
        --out-csv   "$FS_LMDB_CSV" \
        $( [ -n "$FS_MAP_SIZE_GB" ] && echo "--map-size-gb $FS_MAP_SIZE_GB" ) \
        $( [ -n "$LMDB_WORKERS" ]   && echo "--num-workers $LMDB_WORKERS" ) \
        $( [ "$ALLOW_ERRORS" = true ] && echo "--allow-errors" )

    # ── Stage 0b: CTC pre-training ───────────────────────────────────────────
    echo ""
    echo "[Pretrain] CTC pre-training on ASL Fingerspelling..."
    mkdir -p "$PRETRAIN_CHECKPOINT_DIR"
    uv run python -m cnn_transformer.pretrain_fingerspelling \
        --data-dir   "$FS_DATA_DIR" \
        --lmdb-path  "$FS_LMDB_PATH" \
        --lmdb-csv   "$FS_LMDB_CSV" \
        --out-dir    "$PRETRAIN_CHECKPOINT_DIR" \
        --epochs     "$PRETRAIN_EPOCHS" \
        --num-workers "$NUM_WORKERS"

    PRETRAINED_BACKBONE="$PRETRAIN_CHECKPOINT_DIR/backbone_best.pth"
    echo ""
    echo "[Pretrain] Backbone saved → $PRETRAINED_BACKBONE"
fi

# ── Stage 1: ASL LMDB ────────────────────────────────────────────────────────
if [ "$SKIP_LMDB" = false ]; then
    echo ""
    echo "[LMDB] Building / resuming ASL LMDB archive (existing keys are skipped)..."
    uv run python -m cnn_transformer.data.build_lmdb \
        --data-dir  "$DATA_DIR" \
        --lmdb-path "$LMDB_PATH" \
        $( [ -n "$MAP_SIZE_GB" ]   && echo "--map-size-gb $MAP_SIZE_GB" ) \
        $( [ -n "$LMDB_WORKERS" ] && echo "--num-workers $LMDB_WORKERS" ) \
        $( [ "$ALLOW_ERRORS" = true ] && echo "--allow-errors" )
else
    echo ""
    echo "[LMDB] Skipped (--skip-lmdb). Using pre-built LMDB at $LMDB_PATH."
fi

# ── Stage 2: Fine-tuning ──────────────────────────────────────────────────────
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
    --num-workers "$NUM_WORKERS" \
    ${PRETRAINED_BACKBONE:+--pretrained-backbone "$PRETRAINED_BACKBONE"} \
    $( [ "$COMPILE" = true ] && echo "--compile" )

echo ""
echo "========================================"
echo "Training complete. Best model: $CHECKPOINT_DIR/best_final.pth"
echo "========================================"
