#!/bin/bash
# LandmarkConformer training pipeline
# Stages:
#   0. Fingerspelling LMDB build + CTC pre-training (optional)
#   1. ASL LMDB build
#   2. Fine-tuning (with backbone from stage 0 if available)
#
# Usage:
#   bash scripts/run_pipeline_isolated.sh
#   bash scripts/run_pipeline_isolated.sh --skip-pretrain               # skip FS LMDB + pre-training
#   bash scripts/run_pipeline_isolated.sh --skip-fs-lmdb                # skip FS LMDB build but still pre-train
#   bash scripts/run_pipeline_isolated.sh --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth
#   bash scripts/run_pipeline_isolated.sh --phase1-epochs 10 --phase2-epochs 5  # quick test
#   bash scripts/run_pipeline_isolated.sh --map-size-gb 200
#   bash scripts/run_pipeline_isolated.sh --allow-errors
#   bash scripts/run_pipeline_isolated.sh --val-fold 0              # signer fold 0 of 7 (CV)
#   bash scripts/run_pipeline_isolated.sh --stretch-mode sample --stretch-min 0.5 --stretch-max 2.0 --stretch-prob 0.8
#   bash scripts/run_pipeline_isolated.sh --max-frames 256           # keep long clips longer (default 128)
#   bash scripts/run_pipeline_isolated.sh --hand-drop-prob 0.5       # simulate hand-tracking gaps (default off)
#   bash scripts/run_pipeline_isolated.sh --loss ce --seed 1         # loss ablation / another seed
#   bash scripts/run_pipeline_isolated.sh --mixup-prob 0 / --finger-drop-prob 0 / --zero-parts face / --no-depth   # ablations
#   bash scripts/run_pipeline_isolated.sh --supcon-weight 0.2        # cross-signer contrastive loss
#   bash scripts/run_pipeline_isolated.sh --grl-lambda 0             # GRL off (also --dropout, --drop-path-max)
#   bash scripts/run_pipeline_isolated.sh --aug-rotate 30 --aug-shear 0.15 --aug-scale 0.2 --aug-affine-prob 0.75
#   bash scripts/run_pipeline_isolated.sh --arch graph               # GraphConformer: hand-skeleton graph encoder
#   bash scripts/run_pipeline_isolated.sh --arch graph --bone-aug-prob 0.5   # + bone-length (hand proportion) augmentation
#   bash scripts/run_pipeline_isolated.sh --arch graph --val-fold 0  # FS pre-train (same arch/seed) then fine-tune
#   bash scripts/run_pipeline_isolated.sh --arch graph --fs-cleanup  # ... and delete the FS LMDB (~16 GiB) after pre-training
#
# Fingerspelling data is fetched only when pre-training actually runs (not with
# --skip-pretrain or --pretrained-backbone): if the FS LMDB is not at its default
# path, scripts/setup_runpod.sh --with-fs downloads it from shravnchandr/asl-fs-lmdb and
# verifies it, after checking there is enough free disk.
#
# Recommended (downloaded LMDB datasets, skip all local builds):
#   bash scripts/run_pipeline_isolated.sh --skip-pretrain
#   bash scripts/run_pipeline_isolated.sh --pretrain-epochs 30   # pre-train longer (default 20, patience 5)
#
# tmux: started outside tmux, the pipeline relaunches itself in a detached tmux
# session (default name "islr") so it survives a closed terminal / browser tab,
# and tees all output to logs/isolated_<timestamp>.log.
#   tmux attach -t islr        # watch live; detach again with Ctrl-b then d
#   tail -f logs/isolated_*.log
#   bash scripts/run_pipeline_isolated.sh --no-tmux ...   # run in the foreground
#   bash scripts/run_pipeline_isolated.sh --tmux-session exp2 ...  # parallel/other run

set -e
ORIG_ARGS=("$@")

# Run from the repo root regardless of the caller's cwd (relative paths below
# and in arguments are repo-relative).
cd "$(dirname "$0")/.."
SELF="$(pwd)/$(basename "$0")"

# scripts/setup_runpod.sh installs uv into ~/.local/bin, which is only on PATH in shells
# started afterwards — not the terminal that ran setup, nor a tmux session
# launched from it. Exported here, it also reaches the tmux child.
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv not found — run scripts/setup_runpod.sh first." >&2; exit 1; }

export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Python block-buffers stdout into a pipe; without this the tee'd log and the
# tmux pane lag by minutes.
export PYTHONUNBUFFERED=1

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR="data/asl-is-lmdb"          # downloaded from shravnchandr/asl-is-lmdb
CACHE_DIR="data/cache/islr"
LMDB_PATH="data/asl-is-lmdb/is.lmdb.mdb"
CHECKPOINT_DIR="checkpoints/islr"
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
PRETRAIN_EXTRA_ARGS=()  # forwarded to pretrain_fingerspelling.py (arch / seed)
USE_TMUX=true
TMUX_SESSION="islr"

# Fingerspelling pre-training
FS_DATA_DIR="data/asl-fs-lmdb"       # downloaded from shravnchandr/asl-fs-lmdb
FS_LMDB_PATH="data/asl-fs-lmdb/fs.lmdb.mdb"
FS_LMDB_CSV="data/asl-fs-lmdb/train.csv"
PRETRAIN_CHECKPOINT_DIR="checkpoints/pretrain_fs"
PRETRAIN_EPOCHS=20
PRETRAIN_PATIENCE=5
PRETRAINED_BACKBONE=""
SKIP_PRETRAIN=false
SKIP_FS_LMDB=true  # FS LMDB pre-built (downloaded on demand); false = build from raw parquets
FS_MAP_SIZE_GB=""   # empty = 1 TiB default
FS_CLEANUP=false    # delete the FS LMDB after pre-training to free ~16 GiB
FS_DEFAULT_LMDB="$FS_LMDB_PATH"

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
        --fs-cleanup)              FS_CLEANUP=true;              shift ;;
        --compile)                 COMPILE=true;                  shift ;;
        --backbone-warmup-epochs)  BACKBONE_WARMUP_EPOCHS="$2";  shift 2 ;;
        --backbone-lr-ratio)       BACKBONE_LR_RATIO="$2";       shift 2 ;;
        --val-fold)                VAL_FOLD="$2";                shift 2 ;;
        --n-folds)                 N_FOLDS="$2";                 shift 2 ;;
        --stretch-mode|--stretch-min|--stretch-max|--stretch-prob|--max-frames|--hand-drop-prob|--hand-drop-min|--hand-drop-max|--train-eval-size|--loss|--mixup-prob|--finger-drop-prob|--zero-parts|--supcon-weight|--supcon-temp|--supcon-queue|--supcon-dim|--aug-rotate|--aug-shear|--aug-scale|--aug-affine-prob|--grl-lambda|--dropout|--drop-path-max|--bone-aug-prob|--bone-aug-hand|--bone-aug-bone|--vq-codes|--vq-beta|--vq-decay)
                                   TRAIN_EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        # Architecture and seed must match between pre-training and fine-tuning.
        --arch|--graph-hidden|--graph-layers|--seed)
                                   TRAIN_EXTRA_ARGS+=("$1" "$2"); PRETRAIN_EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
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
    LOG="logs/isolated_$(date +%Y%m%d_%H%M%S).log"
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
    echo "  FS LMDB cleanup:    $FS_CLEANUP"
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
        uv run python -m islr.core.data.build_fingerspelling_lmdb \
            --data-dir  "$FS_DATA_DIR" \
            --lmdb-path "$FS_LMDB_PATH" \
            --out-csv   "$FS_LMDB_CSV" \
            $( [ -n "$FS_MAP_SIZE_GB" ] && echo "--map-size-gb $FS_MAP_SIZE_GB" ) \
            $( [ -n "$LMDB_WORKERS" ]   && echo "--num-workers $LMDB_WORKERS" ) \
            $( [ "$ALLOW_ERRORS" = true ] && echo "--allow-errors" )
    elif [ "$FS_LMDB_PATH" = "$FS_DEFAULT_LMDB" ]; then
        # Default location: verify it, or download it (only now that it is needed).
        echo ""
        echo "[FS LMDB] Verifying / downloading the pre-built fingerspelling LMDB..."
        bash scripts/setup_runpod.sh --skip-env --with-fs
    elif [ -f "$FS_LMDB_PATH" ]; then
        echo ""
        echo "[FS LMDB] Using pre-built LMDB at $FS_LMDB_PATH."
    else
        echo "FS LMDB not found at $FS_LMDB_PATH (custom path — not downloaded automatically)." >&2
        exit 1
    fi

    # ── Stage 0b: CTC pre-training ───────────────────────────────────────────
    echo ""
    echo "[Pretrain] CTC pre-training on ASL Fingerspelling..."
    mkdir -p "$PRETRAIN_CHECKPOINT_DIR"
    uv run python -m islr.pretrain.fingerspelling \
        --data-dir   "$FS_DATA_DIR" \
        --lmdb-path  "$FS_LMDB_PATH" \
        --lmdb-csv   "$FS_LMDB_CSV" \
        --out-dir    "$PRETRAIN_CHECKPOINT_DIR" \
        --epochs     "$PRETRAIN_EPOCHS" \
        --patience   "$PRETRAIN_PATIENCE" \
        --num-workers "$NUM_WORKERS" \
        "${PRETRAIN_EXTRA_ARGS[@]}"

    PRETRAINED_BACKBONE="$PRETRAIN_CHECKPOINT_DIR/backbone_best.pth"
    echo ""
    echo "[Pretrain] Backbone saved → $PRETRAINED_BACKBONE"

    if [ "$FS_CLEANUP" = true ]; then
        echo "[FS LMDB] --fs-cleanup: deleting $FS_LMDB_PATH (fine-tuning does not need it)"
        rm -f "$FS_LMDB_PATH"
    fi
fi

# ── Stage 1: ASL LMDB ────────────────────────────────────────────────────────
if [ "$SKIP_LMDB" = false ]; then
    echo ""
    echo "[LMDB] Building / resuming ASL LMDB archive (existing keys are skipped)..."
    uv run python -m islr.core.data.build_lmdb \
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
uv run python -m islr.isolated.train \
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
