#!/bin/bash
# LandmarkConformer training pipeline
# Stages:
#   0. Fingerspelling LMDB build + CTC pre-training (optional)
#   1. ASL LMDB build
#   2. Fine-tuning (with backbone from stage 0 if available)
#
# Usage:
#   bash run_pipeline_cnn_transformer.sh
#   bash run_pipeline_cnn_transformer.sh --skip-pretrain               # skip FS LMDB + pre-training
#   bash run_pipeline_cnn_transformer.sh --skip-fs-lmdb                # skip FS LMDB build but still pre-train
#   bash run_pipeline_cnn_transformer.sh --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth
#   bash run_pipeline_cnn_transformer.sh --phase1-epochs 10 --phase2-epochs 5  # quick test
#   bash run_pipeline_cnn_transformer.sh --map-size-gb 200
#   bash run_pipeline_cnn_transformer.sh --allow-errors
#   bash run_pipeline_cnn_transformer.sh --val-fold 0              # signer fold 0 of 7 (CV)
#   bash run_pipeline_cnn_transformer.sh --stretch-mode sample --stretch-min 0.5 --stretch-max 2.0 --stretch-prob 0.8
#   bash run_pipeline_cnn_transformer.sh --max-frames 256           # keep long clips longer (default 128)
#   bash run_pipeline_cnn_transformer.sh --hand-drop-prob 0.5       # simulate hand-tracking gaps (default off)
#   bash run_pipeline_cnn_transformer.sh --loss ce --seed 1         # loss ablation / another seed
#   bash run_pipeline_cnn_transformer.sh --mixup-prob 0 / --finger-drop-prob 0 / --zero-parts face / --no-depth   # ablations
#
# Recommended (downloaded LMDB datasets, skip all local builds):
#   bash run_pipeline_cnn_transformer.sh --skip-pretrain
#   bash run_pipeline_cnn_transformer.sh --skip-fs-lmdb --pretrain-epochs 40  # pre-train from downloaded FS LMDB
#
# tmux: started outside tmux, the pipeline relaunches itself in a detached tmux
# session (default name "islr") so it survives a closed terminal / browser tab,
# and tees all output to logs/cnn_transformer_<timestamp>.log.
#   tmux attach -t islr        # watch live; detach again with Ctrl-b then d
#   tail -f logs/cnn_transformer_*.log
#   bash run_pipeline_cnn_transformer.sh --no-tmux ...   # run in the foreground
#   bash run_pipeline_cnn_transformer.sh --tmux-session exp2 ...  # parallel/other run

set -e
ORIG_ARGS=("$@")

# Run from the repo root regardless of the caller's cwd (relative paths below
# and in arguments are repo-relative).
cd "$(dirname "$0")"
SELF="$(pwd)/$(basename "$0")"

# setup_runpod.sh installs uv into ~/.local/bin, which is only on PATH in shells
# started afterwards — not the terminal that ran setup, nor a tmux session
# launched from it. Exported here, it also reaches the tmux child.
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv not found — run setup_runpod.sh first." >&2; exit 1; }

export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Python block-buffers stdout into a pipe; without this the tee'd log and the
# tmux pane lag by minutes.
export PYTHONUNBUFFERED=1

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR="data/asl-is-lmdb"          # downloaded from shravnchandr/asl-is-lmdb
CACHE_DIR="data/cache/cnn_transformer"
LMDB_PATH="data/asl-is-lmdb/is.lmdb.mdb"
CHECKPOINT_DIR="checkpoints/cnn_transformer"
PHASE1_EPOCHS=80
PHASE2_EPOCHS=0    # cosine warmdown; never beat Phase 1 best in Runs 002/003
PATIENCE=20
BATCH_SIZE=64
NUM_WORKERS=4
SKIP_LMDB=true   # LMDB pre-built; set false only when building from raw parquets
MAP_SIZE_GB=""   # empty = 1 TiB default (sparse file, no disk cost)
ALLOW_ERRORS=false
LMDB_WORKERS=4
COMPILE=false
BACKBONE_WARMUP_EPOCHS=5   # epochs to freeze backbone after loading pretrained weights
BACKBONE_LR_RATIO=0.1       # backbone LR as fraction of head LR after warmup
VAL_FOLD=""        # empty = default split (Runs 001–005); 0..N_FOLDS-1 = signer fold
N_FOLDS=7
TRAIN_EXTRA_ARGS=()  # forwarded to train.py only when set (stretch / max-frames; train.py defaults otherwise)
USE_TMUX=true
TMUX_SESSION="islr"

# Fingerspelling pre-training
FS_DATA_DIR="data/asl-fs-lmdb"       # downloaded from shravnchandr/asl-fs-lmdb
FS_LMDB_PATH="data/asl-fs-lmdb/fs.lmdb.mdb"
FS_LMDB_CSV="data/asl-fs-lmdb/train.csv"
PRETRAIN_CHECKPOINT_DIR="checkpoints/pretrain_fs"
PRETRAIN_EPOCHS=40
PRETRAIN_PATIENCE=10
PRETRAINED_BACKBONE=""
SKIP_PRETRAIN=false
SKIP_FS_LMDB=true  # FS LMDB pre-built; set false only when building from raw parquets
FS_MAP_SIZE_GB=""   # empty = 1 TiB default

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
        --build-lmdb)              SKIP_LMDB=false;              shift ;;
        --map-size-gb)             MAP_SIZE_GB="$2";             shift 2 ;;
        --allow-errors)            ALLOW_ERRORS=true;            shift ;;
        --lmdb-workers)            LMDB_WORKERS="$2";            shift 2 ;;
        --fs-data-dir)             FS_DATA_DIR="$2";             shift 2 ;;
        --fs-lmdb-path)            FS_LMDB_PATH="$2";            shift 2 ;;
        --fs-lmdb-csv)             FS_LMDB_CSV="$2";             shift 2 ;;
        --pretrain-checkpoint-dir) PRETRAIN_CHECKPOINT_DIR="$2"; shift 2 ;;
        --pretrain-epochs)         PRETRAIN_EPOCHS="$2";         shift 2 ;;
        --pretrain-patience)       PRETRAIN_PATIENCE="$2";       shift 2 ;;
        --pretrained-backbone)     PRETRAINED_BACKBONE="$2";     shift 2 ;;
        --skip-pretrain)           SKIP_PRETRAIN=true;           shift ;;
        --skip-fs-lmdb)            SKIP_FS_LMDB=true;            shift ;;
        --build-fs-lmdb)           SKIP_FS_LMDB=false;           shift ;;
        --fs-map-size-gb)          FS_MAP_SIZE_GB="$2";          shift 2 ;;
        --compile)                 COMPILE=true;                  shift ;;
        --backbone-warmup-epochs)  BACKBONE_WARMUP_EPOCHS="$2";  shift 2 ;;
        --backbone-lr-ratio)       BACKBONE_LR_RATIO="$2";       shift 2 ;;
        --val-fold)                VAL_FOLD="$2";                shift 2 ;;
        --n-folds)                 N_FOLDS="$2";                 shift 2 ;;
        --stretch-mode|--stretch-min|--stretch-max|--stretch-prob|--max-frames|--hand-drop-prob|--hand-drop-min|--hand-drop-max|--seed|--train-eval-size|--loss|--mixup-prob|--finger-drop-prob|--zero-parts)
                                   TRAIN_EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --no-depth)                TRAIN_EXTRA_ARGS+=("$1");   shift ;;
        --no-tmux)                 USE_TMUX=false;               shift ;;
        --tmux-session)            TMUX_SESSION="$2";            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── tmux: relaunch detached so the run survives a closed terminal ───────────
if [ "$USE_TMUX" = true ] && [ -z "${TMUX:-}" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux not found. Install it (apt-get install -y tmux) or pass --no-tmux." >&2
        exit 1
    fi
    if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists — a run may be in progress." >&2
        echo "  attach:  tmux attach -t $TMUX_SESSION" >&2
        echo "  or start another with --tmux-session <name>, or end it: tmux kill-session -t $TMUX_SESSION" >&2
        exit 1
    fi
    mkdir -p logs
    LOG="logs/cnn_transformer_$(date +%Y%m%d_%H%M%S).log"
    # Child runs with --no-tmux (no recursion). The pane stays open afterwards
    # (exec bash) so the final output and exit code can still be inspected.
    RUN_CMD=$(printf '%q ' bash "$SELF" "${ORIG_ARGS[@]}" --no-tmux)
    INNER="set -o pipefail; $RUN_CMD 2>&1 | tee $(printf '%q' "$LOG"); code=\$?; echo; echo \"[pipeline exited with code \$code — log: $LOG]\"; exec bash"
    tmux new-session -d -s "$TMUX_SESSION" -c "$(pwd)" bash -c "$INNER"
    echo "Started in tmux session '$TMUX_SESSION'. It keeps running if you disconnect."
    echo "  watch live:  tmux attach -t $TMUX_SESSION     (detach: Ctrl-b then d)"
    echo "  log file:    tail -f $LOG"
    echo "  stop run:    tmux kill-session -t $TMUX_SESSION"
    exit 0
fi

# If --pretrained-backbone is given directly, skip the pre-training stage
if [ -n "$PRETRAINED_BACKBONE" ]; then
    SKIP_PRETRAIN=true
fi

mkdir -p "$CACHE_DIR" "$CHECKPOINT_DIR"

echo "========================================"
echo "LandmarkConformer Pipeline"
echo "  Data dir:           $DATA_DIR"
echo "  Cache dir:          $CACHE_DIR"
echo "  LMDB path:          $LMDB_PATH"
echo "  Checkpoint dir:     $CHECKPOINT_DIR"
echo "  Phase 1 epochs:     $PHASE1_EPOCHS"
echo "  Phase 2 epochs:     $PHASE2_EPOCHS"
echo "  Patience:           $PATIENCE"
echo "  Batch size:         $BATCH_SIZE"
echo "  Num workers:        $NUM_WORKERS"
if [ -n "$VAL_FOLD" ]; then echo "  Validation:         fold $VAL_FOLD/$N_FOLDS"; else echo "  Validation:         default split"; fi
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
    if [ "$SKIP_FS_LMDB" = false ]; then
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
    else
        echo ""
        echo "[FS LMDB] Skipped (--skip-fs-lmdb). Using pre-built LMDB at $FS_LMDB_PATH."
    fi

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
        --patience   "$PRETRAIN_PATIENCE" \
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
echo "[Train] Running two-phase LandmarkConformer training..."
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
    --backbone-warmup-epochs "$BACKBONE_WARMUP_EPOCHS" \
    --backbone-lr-ratio "$BACKBONE_LR_RATIO" \
    ${VAL_FOLD:+--val-fold "$VAL_FOLD"} \
    --n-folds "$N_FOLDS" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    $( [ "$COMPILE" = true ] && echo "--compile" )

echo ""
echo "========================================"
# best_final.pth only exists if Phase 2 ran and beat the Phase 1 best.
if [ -f "$CHECKPOINT_DIR/best_final.pth" ]; then BEST="$CHECKPOINT_DIR/best_final.pth"; else BEST="$CHECKPOINT_DIR/best_phase1.pth"; fi
echo "Training complete. Best model: $BEST"
echo "========================================"
