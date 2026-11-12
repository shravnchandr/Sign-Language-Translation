"""Extract MediaPipe Holistic landmarks from ASL Citizen videos on Kaggle (CPU), sharded.

RESEARCH USE ONLY: ASL Citizen is under the Microsoft Research License Terms
(non-commercial research; no redistribution of the data or modifications). Kernels and
their outputs stay private; models trained on it are not exported for the app.

Each shard is its own private notebook `<user>/extract-aslcitizen-s<k>` so shards run in
parallel sessions. A shard takes every N-th video (sorted by file name) among the clips
whose gloss is in `aslcitizen_gloss_map.json` (or all clips with --all), and writes the
competition format used by ASL Signs / WLASL — one parquet per video with columns frame,
type, landmark_index, x, y, z (NaN when a part is not detected) — plus shard_<k>.csv
(path, participant, split, gloss, video, n_frames). Same MediaPipe version, settings and
pinned Kaggle image as `preprocess-wlasl-mediapipe`.

Usage (from the repo root; the repo is cloned at the pushed HEAD):
  python scripts/extract_aslcitizen.py push --shards 1 --limit 20 --name extract-aslcitizen-smoke
  python scripts/extract_aslcitizen.py push --shards 3
  python scripts/extract_aslcitizen.py status --shards 3
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kaggle_run import REPO_URL, ROOT, kaggle, user  # noqa: E402

DATASET = "abd0kamel/asl-citizen"
# Same image as preprocess-wlasl-mediapipe (mediapipe 0.10.14 needs Python <= 3.12).
DOCKER = "gcr.io/kaggle-images/python@sha256:2565f241bd4ecac85a132676664dabd5e3e8b4567db1b936507a987b37095f59"

CELL_SETUP = """
import subprocess, sys, os, glob, json, time
subprocess.run("pip install -q mediapipe==0.10.14 opencv-python-headless pyarrow", shell=True, check=True)
subprocess.run(f"git clone -q {REPO_URL} /tmp/repo && git -C /tmp/repo checkout -q {REF}", shell=True, check=True)
import mediapipe as mp, pandas as pd, numpy as np, cv2
print("mediapipe", mp.__version__, "| python", sys.version.split()[0], "| CPUs", os.cpu_count())
"""

CELL_EXTRACT = """
root = os.path.dirname(sorted(glob.glob("/kaggle/input/**/ASL_Citizen/splits", recursive=True))[0])
splits = pd.concat([pd.read_csv(f"{root}/splits/{s}.csv").assign(split=s) for s in ("train", "val", "test")])
gmap = json.load(open("/tmp/repo/research/models/islr/core/data/aslcitizen_gloss_map.json"))
if not ALL:
    splits = splits[splits["Gloss"].isin(gmap)]
splits = splits.sort_values("Video file").reset_index(drop=True)
mine = splits.iloc[SHARD::SHARDS]
if LIMIT:
    mine = mine.head(LIMIT)
print(f"{len(splits)} clips selected, shard {SHARD}/{SHARDS}: {len(mine)}")
OUT = "/kaggle/working/aslcitizen_landmarks"
os.makedirs(f"{OUT}/train_landmark_files", exist_ok=True)

PARTS = (("pose", 33, "pose_landmarks"), ("left_hand", 21, "left_hand_landmarks"),
         ("right_hand", 21, "right_hand_landmarks"), ("face", 468, "face_landmarks"))

def init():
    global holistic
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)

N_LM = sum(n for _, n, _ in PARTS)
TYPES = np.concatenate([[part] * n for part, n, _ in PARTS])
INDEX = np.concatenate([np.arange(n) for _, n, _ in PARTS])

def extract(job):
    video, out = job
    if os.path.exists(out):
        return out, pd.read_parquet(out, columns=["frame"])["frame"].nunique()
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        xyz = np.full((N_LM, 3), np.nan, dtype=np.float32)
        i = 0
        for part, n, attr in PARTS:
            lms = getattr(res, attr)
            if lms:
                xyz[i:i + n] = [[l.x, l.y, l.z] for l in lms.landmark[:n]]
            i += n
        frames.append(xyz)
    cap.release()
    if not frames:
        return out, 0
    T = len(frames)
    xyz = np.concatenate(frames)
    df = pd.DataFrame({"frame": np.repeat(np.arange(T), N_LM), "type": np.tile(TYPES, T),
                       "landmark_index": np.tile(INDEX, T), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    df["row_id"] = df["frame"].astype(str) + "-" + df["type"] + "-" + df["landmark_index"].astype(str)
    df.to_parquet(out, index=False)
    return out, T

from multiprocessing import Pool
jobs = [(f"{root}/videos/{v}", f"{OUT}/train_landmark_files/{os.path.splitext(v)[0]}.parquet")
        for v in mine["Video file"]]
t0, done = time.time(), []
with Pool(os.cpu_count(), initializer=init) as pool:
    for i, r in enumerate(pool.imap(extract, jobs, chunksize=4)):
        done.append(r)
        if i % 200 == 0:
            print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
nf = dict(done)
mine = mine.assign(path=[os.path.relpath(o, OUT) for _, o in jobs], n_frames=[nf[o] for _, o in jobs])
mine = mine.rename(columns={"Participant ID": "participant", "Gloss": "gloss", "Video file": "video"})
mine = mine[mine["n_frames"] > 0]
mine[["path", "participant", "split", "gloss", "video", "n_frames"]].to_csv(f"{OUT}/shard_{SHARD}.csv", index=False)
print(f"done: {len(mine)} clips in {(time.time() - t0) / 3600:.2f} h; frames median {mine['n_frames'].median()}")
"""


def notebook(ref: str, shard: int, shards: int, all_clips: bool, limit: int = 0) -> dict:
    def code(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.strip("\n").splitlines(True)}

    head = (f"REF, SHARD, SHARDS, ALL, LIMIT, REPO_URL = {ref!r}, {shard}, {shards}, {all_clips}, "
            f"{limit}, {REPO_URL!r}\n")
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [f"ASL Citizen → MediaPipe landmarks, shard {shard}/{shards} — **research use only** "
                        "(Microsoft Research License Terms). Generated by `scripts/extract_aslcitizen.py`."]},
            code(head + CELL_SETUP), code(CELL_EXTRACT),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["push", "status"])
    p.add_argument("--shards", type=int, default=3)
    p.add_argument("--only", type=int, nargs="*", help="shard numbers (default: all)")
    p.add_argument("--all", action="store_true", help="every ASL Citizen clip, not only mapped glosses")
    p.add_argument("--limit", type=int, default=0, help="first N videos per shard (smoke test)")
    p.add_argument("--name", default="extract-aslcitizen", help="kernel name prefix")
    a = p.parse_args()
    shards = a.only if a.only else range(a.shards)
    if a.action == "status":
        for k in shards:
            print(k, kaggle("kernels", "status", f"{user()}/{a.name}-s{k}", check=False).strip())
        return
    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    if not subprocess.run(["git", "-C", str(ROOT), "branch", "-r", "--contains", ref],
                          capture_output=True, text=True).stdout.strip():
        sys.exit(f"HEAD {ref[:7]} is not pushed — the notebook clones from GitHub. Push first.")
    for k in shards:
        name = f"{a.name}-s{k}"
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"{name}.ipynb").write_text(json.dumps(notebook(ref, k, a.shards, a.all, a.limit), indent=1))
            meta = {"id": f"{user()}/{name}", "title": name, "code_file": f"{name}.ipynb",
                    "language": "python", "kernel_type": "notebook", "is_private": True,
                    "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [],
                    "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
                    "model_sources": [], "docker_image": DOCKER}
            (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            print(k, kaggle("kernels", "push", "-p", tmp).strip())


if __name__ == "__main__":
    main()
