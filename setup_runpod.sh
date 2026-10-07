#!/bin/bash
# One-shot setup for a fresh RunPod pod with no persistent volume.
#
# Installs the environment, downloads the pre-built ASL Signs LMDB from the
# saved output of the Kaggle build notebook, and lays it out exactly where
# run_pipeline_cnn_transformer.sh / train.py read it by default:
#
#   data/asl-is-lmdb/  is.lmdb.mdb  train.csv  sign_to_prediction_index_map.json
#
# Kernel output is fetched by research/tools/download_kernel_output.py rather than
# `kaggle kernels output`, which buffers each file fully in RAM (OOM risk on a
# 15–20 GB LMDB). Downloads land in data/.kaggle_download/ — same filesystem as
# data/asl-is-lmdb/, so the final move is an instant rename (peak disk ≈ 1× LMDB).
#
# Any output layout is accepted (nested folders, flat *.mdb, or an LMDB
# directory holding data.mdb) and normalised to the above. The archive is then
# verified with the training code's own key scheme, so a stale LMDB fails here
# rather than after training starts.
#
# Kaggle credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY.
#
# Usage (from the repo root):
#   bash setup_runpod.sh                                         # build-islt-lmdb notebook output
#   bash setup_runpod.sh --kernel <owner>/<notebook-slug>        # a different notebook
#   bash setup_runpod.sh --dataset shravnchandr/asl-is-lmdb      # from a Kaggle dataset instead
#   bash setup_runpod.sh --from /path/to/downloaded/dir          # files already on disk
#   bash setup_runpod.sh --skip-env                              # data only
#   bash setup_runpod.sh --force                                 # re-download even if verified

set -euo pipefail
cd "$(dirname "$0")"

# ── Defaults ────────────────────────────────────────────────────────────────
KERNEL="shravnchandr/build-islt-lmdb"
DATASET=""
FROM=""
SKIP_ENV=false
FORCE=false
DEST="data/asl-is-lmdb"
DL_DIR="data/.kaggle_download"   # same filesystem as $DEST → mv is a rename; removed after

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernel)   KERNEL="$2";  DATASET=""; shift 2 ;;
        --dataset)  DATASET="$2"; KERNEL="";  shift 2 ;;
        --from)     FROM="$2";    shift 2 ;;
        --skip-env) SKIP_ENV=true; shift ;;
        --force)    FORCE=true;    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"
die() { echo "ERROR: $*" >&2; exit 1; }
free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }

# ── 1. Environment ──────────────────────────────────────────────────────────
if [ "$SKIP_ENV" = false ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[env] Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        # run_pipeline_cnn_transformer.sh runs training inside tmux by default
        echo "[env] Installing tmux..."
        SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux >/dev/null
    fi
    echo "[env] uv sync..."
    uv sync
    echo "[env] Checking CUDA..."
    uv run python -c "
import sys, torch
ok = torch.cuda.is_available()
print(f'  torch {torch.__version__} | CUDA available: {ok}' + (f' | {torch.cuda.get_device_name(0)}' if ok else ''))
sys.exit(0 if ok else 1)" || die "torch cannot see a GPU — training would run on CPU. Check the pod's GPU/driver."
fi
command -v uv >/dev/null 2>&1 || die "uv not found (run without --skip-env)"

# ── Verification: open the LMDB through the training code, check the keys for
#    the first 500 train.csv rows exist and that one sample decodes. ─────────
verify() {
    uv run python - "$DEST" <<'PY'
import io, json, sys
from pathlib import Path
import pandas as pd, torch
from cnn_transformer.config import COORD_FEAT
from cnn_transformer.data.dataset import _open_lmdb_env
from cnn_transformer.data._cache_keys import CACHE_VERSION, lmdb_key

d = Path(sys.argv[1])
df = pd.read_csv(d / "train.csv")
missing = {"path", "participant_id", "sign"} - set(df.columns)
assert not missing, f"train.csv is missing columns {missing}"
n_signs = len(json.load(open(d / "sign_to_prediction_index_map.json")))

with _open_lmdb_env(str(d / "is.lmdb.mdb")).begin() as txn:
    vals = [txn.get(lmdb_key(p)) for p in df["path"][:500]]
    hits = sum(v is not None for v in vals)
    first = next((bytes(v) for v in vals if v is not None), None)
print(f"  CACHE_VERSION {CACHE_VERSION}: {hits}/{len(vals)} keys found | "
      f"{len(df):,} samples, {df['participant_id'].nunique()} signers, {n_signs} signs")
assert hits == len(vals), (
    "LMDB keys do not match the training code: this archive was built with an older "
    "version. Rebuild it on Kaggle (notebooks/tools/create-is-lmdb.ipynb) and re-run."
)
x = torch.load(io.BytesIO(first), weights_only=True)
assert x.ndim == 2 and x.shape[1] == COORD_FEAT, f"sample shape {tuple(x.shape)}, expected (T, {COORD_FEAT})"
print(f"  sample decodes OK: {tuple(x.shape)}")
PY
}

# ── 2. Data ─────────────────────────────────────────────────────────────────
if [ "$FORCE" = false ] && [ -f "$DEST/is.lmdb.mdb" ] && verify 2>/dev/null; then
    echo "[data] Already present and verified in $DEST — skipping download (--force to redo)."
else
    if [ -n "$FROM" ]; then
        SRC="$FROM"
        echo "[data] Using already-downloaded files in $SRC"
    else
        [ -f "$HOME/.kaggle/kaggle.json" ] || [ -f "$HOME/.config/kaggle/kaggle.json" ] \
            || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; } \
            || die "No Kaggle credentials. Upload kaggle.json (kaggle.com → Settings → API → Create New Token), then:
       mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     or export KAGGLE_USERNAME and KAGGLE_KEY."
        [ -f "$HOME/.kaggle/kaggle.json" ] && chmod 600 "$HOME/.kaggle/kaggle.json"

        SRC="$DL_DIR"
        # Kernel downloads keep $SRC so an interrupted run resumes its .part
        # files; the dataset zip can't resume, so that route starts clean.
        [ -n "$KERNEL" ] || rm -rf "$SRC"
        mkdir -p "$SRC"
        echo "[data] Free disk: $(free_gb "$SRC") GB"
        if [ -n "$KERNEL" ]; then
            echo "[data] Downloading saved output of notebook $KERNEL ..."
            uv run python research/tools/download_kernel_output.py "$KERNEL" -p "$SRC" \
                --include '*.mdb' train.csv sign_to_prediction_index_map.json \
                --exclude lock.mdb
        else
            # --unzip keeps zip + extracted on disk until it finishes (~2× size peak)
            echo "[data] Downloading dataset $DATASET ..."
            uv run kaggle datasets download "$DATASET" -p "$SRC" --unzip
        fi
    fi

    # Locate the pieces wherever the download put them. The largest *.mdb that
    # isn't a lock file is the archive (flat is.lmdb.mdb or dir-form data.mdb);
    # a renamed data.mdb opens fine as a flat LMDB.
    MDB=$(find "$SRC" -type f -name '*.mdb' ! -name 'lock.mdb' -exec ls -S {} + 2>/dev/null | head -1)
    CSV=$(find "$SRC" -type f -name 'train.csv' | head -1)
    JSON=$(find "$SRC" -type f -name 'sign_to_prediction_index_map.json' | head -1)
    [ -n "$MDB" ]  || die "no LMDB (*.mdb) found under $SRC"
    [ -n "$CSV" ]  || die "train.csv not found under $SRC — copy it from the asl-signs competition data"
    [ -n "$JSON" ] || die "sign_to_prediction_index_map.json not found under $SRC — copy it from the competition data"

    mkdir -p "$DEST"
    if [ -n "$FROM" ]; then  # never move the user's own files
        cp "$MDB" "$DEST/is.lmdb.mdb"; cp "$CSV" "$DEST/train.csv"; cp "$JSON" "$DEST/"
    else
        mv "$MDB" "$DEST/is.lmdb.mdb"; mv "$CSV" "$DEST/train.csv"; mv "$JSON" "$DEST/"
        rm -rf "$SRC"
    fi
    echo "[data] Saved to $DEST ($(du -sh "$DEST/is.lmdb.mdb" | cut -f1) LMDB)"
    echo "[data] Verifying against the training code..."
    verify || die "verification failed (see above)"
fi

echo ""
echo "========================================"
echo "Setup complete. Free disk: $(free_gb .) GB"
echo "Train (runs in a detached tmux session 'islr'; safe to disconnect):"
echo "  bash run_pipeline_cnn_transformer.sh --skip-pretrain --num-workers 8"
echo "  tmux attach -t islr          # watch live; detach with Ctrl-b then d"
echo "  tail -f logs/cnn_transformer_*.log"
echo "========================================"
