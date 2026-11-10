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
#   bash setup_runpod.sh --with-fs                               # also the fingerspelling LMDB
#                                                                #   (dataset shravnchandr/asl-fs-lmdb → data/asl-fs-lmdb/)

set -euo pipefail
CALLER_DIR="$(pwd)"
cd "$(dirname "$0")"
# uv installs to ~/.local/bin, which is only on PATH in shells started after the
# install — pick it up so re-runs in the same terminal don't miss or reinstall it.
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Defaults ────────────────────────────────────────────────────────────────
KERNEL="shravnchandr/build-islt-lmdb"
DATASET=""
FROM=""
SKIP_ENV=false
FORCE=false
DEST="data/asl-is-lmdb"
DL_DIR="data/.kaggle_download"   # same filesystem as $DEST → mv is a rename; removed after
WITH_FS=false
FS_DATASET="shravnchandr/asl-fs-lmdb"
FS_DEST="data/asl-fs-lmdb"
FS_FILES=(character_to_prediction_index.json train.csv fs.lmdb.mdb)
# Peak disk for the FS download: fs.lmdb.mdb is 17.1 GB; if Kaggle serves it zipped,
# the zip and the extracted file coexist until extraction finishes.
FS_NEED_GB=35

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernel)   KERNEL="$2";  DATASET=""; shift 2 ;;
        --dataset)  DATASET="$2"; KERNEL="";  shift 2 ;;
        --from)     FROM="$2";    shift 2 ;;
        --skip-env) SKIP_ENV=true; shift ;;
        --force)    FORCE=true;    shift ;;
        --with-fs)  WITH_FS=true;  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
# --from is the caller's path; resolve it before the cd above changes meaning.
if [ -n "$FROM" ] && [[ "$FROM" != /* ]]; then FROM="$CALLER_DIR/$FROM"; fi

export PYTHONPATH="$(pwd)/research/models${PYTHONPATH:+:$PYTHONPATH}"
die() { echo "ERROR: $*" >&2; exit 1; }
free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }
need_kaggle_creds() {
    [ -f "$HOME/.kaggle/kaggle.json" ] || [ -f "$HOME/.config/kaggle/kaggle.json" ] \
        || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; } \
        || die "No Kaggle credentials. Upload kaggle.json (kaggle.com → Settings → API → Create New Token), then:
       mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     or export KAGGLE_USERNAME and KAGGLE_KEY."
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then chmod 600 "$HOME/.kaggle/kaggle.json"; fi
}

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
        need_kaggle_creds

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

# ── 3. Fingerspelling LMDB (optional, for CTC pre-training) ────────────────
# Keys are fs:{sequence_id} with no preprocessing version, so verification
# also checks the sample width against the current column layout.
verify_fs() {
    uv run python - "$FS_DEST" <<'PY'
import io, sys
from pathlib import Path
import pandas as pd, torch
from cnn_transformer.config import COORD_FEAT
from cnn_transformer.data.dataset import _open_lmdb_env

d = Path(sys.argv[1])
df = pd.read_csv(d / "train.csv")
missing = {"sequence_id", "participant_id", "phrase"} - set(df.columns)
assert not missing, f"train.csv is missing columns {missing}"
assert (d / "character_to_prediction_index.json").exists(), "character_to_prediction_index.json missing"
with _open_lmdb_env(str(d / "fs.lmdb.mdb")).begin() as txn:
    vals = [txn.get(f"fs:{s}".encode()) for s in df["sequence_id"][:500]]
    hits = sum(v is not None for v in vals)
    first = next((bytes(v) for v in vals if v is not None), None)
print(f"  FS LMDB: {hits}/{len(vals)} keys found | {len(df):,} sequences, "
      f"{df['participant_id'].nunique()} signers")
assert hits == len(vals), "FS LMDB keys do not match train.csv"
x = torch.load(io.BytesIO(first), weights_only=True)
assert x.ndim == 2 and x.shape[1] == COORD_FEAT, f"sample shape {tuple(x.shape)}, expected (T, {COORD_FEAT})"
print(f"  sample decodes OK: {tuple(x.shape)}")
PY
}

if [ "$WITH_FS" = true ]; then
    if [ "$FORCE" = false ] && [ -f "$FS_DEST/fs.lmdb.mdb" ] && verify_fs 2>/dev/null; then
        echo "[fs] Already present and verified in $FS_DEST — skipping download (--force to redo)."
    else
        need_kaggle_creds
        FS_SRC="$DL_DIR-fs"
        rm -rf "$FS_SRC"; mkdir -p "$FS_SRC"
        [ -f "$FS_DEST/fs.lmdb.mdb" ] && rm -f "$FS_DEST/fs.lmdb.mdb"  # failed verification: replace
        FREE=$(free_gb "$FS_SRC")
        echo "[fs] Free disk: $FREE GB (need ~$FS_NEED_GB GB: zip + extracted LMDB coexist until unzip ends)"
        [ "$FREE" -ge "$FS_NEED_GB" ] || die "not enough disk for the fingerspelling LMDB ($FREE GB free, ~$FS_NEED_GB GB needed).
     Free space (e.g. old checkpoints, data/cache) or use a bigger volume."
        # File by file: Kaggle returns 404 for the whole-dataset zip of this 16 GiB
        # dataset (2026-11-10), while single-file downloads work.
        for f in "${FS_FILES[@]}"; do
            echo "[fs] Downloading $FS_DATASET/$f ..."
            uv run kaggle datasets download "$FS_DATASET" -f "$f" -p "$FS_SRC"
        done
        # A file may arrive zipped (<name>.zip); extract with Python (no unzip dependency).
        for z in "$FS_SRC"/*.zip; do
            [ -e "$z" ] || continue
            echo "[fs] Extracting $(basename "$z") ..."
            uv run python -m zipfile -e "$z" "$FS_SRC" && rm -f "$z"
        done
        MDB=$(find "$FS_SRC" -type f -name '*.mdb' ! -name 'lock.mdb' -exec ls -S {} + 2>/dev/null | head -1)
        CSV=$(find "$FS_SRC" -type f -name 'train.csv' | head -1)
        JSON=$(find "$FS_SRC" -type f -name 'character_to_prediction_index.json' | head -1)
        [ -n "$MDB" ] && [ -n "$CSV" ] && [ -n "$JSON" ] || die "FS download incomplete under $FS_SRC (need *.mdb, train.csv, character_to_prediction_index.json)"
        mkdir -p "$FS_DEST"
        mv "$MDB" "$FS_DEST/fs.lmdb.mdb"; mv "$CSV" "$FS_DEST/train.csv"; mv "$JSON" "$FS_DEST/"
        rm -rf "$FS_SRC"
        echo "[fs] Saved to $FS_DEST ($(du -sh "$FS_DEST/fs.lmdb.mdb" | cut -f1) LMDB)"
        verify_fs || die "FS verification failed (see above)"
    fi
fi

echo ""
echo "========================================"
echo "Setup complete. Free disk: $(free_gb .) GB"
echo "Train (runs in a detached tmux session 'islr'; safe to disconnect):"
echo "  bash run_pipeline_cnn_transformer.sh --skip-pretrain --num-workers 8"
echo "  tmux attach -t islr          # watch live; detach with Ctrl-b then d"
echo "  tail -f logs/cnn_transformer_*.log"
echo "Before stopping the pod (disk is erased on stop):"
echo "  bash runpod_results.sh send <run-name>     # then 'receive <code>' on your machine"
echo "========================================"
