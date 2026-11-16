"""Selection inventory of an extraction: which videos each shard was given, re-run on Kaggle (CPU).

Extractions before the coverage fix wrote only `shard_<k>.csv`, from which a video that failed to
decode was silently dropped — so a shard's clip list cannot show what is missing. This re-runs the
extraction's own selection cell (CELL_SELECT, taken from the commit the extraction was pinned to,
as the extraction notebooks embed it) for every shard, without extracting anything, and writes
`inventory.csv` (shard, path, video, participant, split, gloss) + `inventory.json` (extractor, ref,
cells digest, counts). Builders take it with `--inventory`; extractions since the fix write
`selected_<k>.csv` themselves.

Usage (from the repo root):
  python scripts/extraction_inventory.py push popsign --ref 07f23b7 --shards 12
  python scripts/extraction_inventory.py push aslcitizen --ref 07f23b7 --shards 3
  python scripts/extraction_inventory.py push wlasl --ref eb9fea3 --shards 4
  python scripts/extraction_inventory.py status popsign
  python scripts/extraction_inventory.py push popsign --ref 07f23b7 --shards 12 --probe-all --name fps-popsign
  then mount `<user>/inventory-<extractor>` and pass --inventory <mount>/inventory.csv
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_aslcitizen import DATASET as ASLC_DATASET, cells_at, cells_digest  # noqa: E402
from extract_popsign import DATASETS as POPSIGN_DATASETS, SPLIT_DATASETS as POPSIGN_SPLITS  # noqa: E402
from extract_wlasl import DATASET as WLASL_DATASET  # noqa: E402
from kaggle_run import REPO_URL, ROOT, kaggle, user  # noqa: E402

# extractor → (script holding CELL_SELECT, datasets, the selection globals its push used)
EXTRACTORS = {
    "popsign": ("extract_popsign.py", POPSIGN_DATASETS, {"PER_PAIR": 4, "SPLIT": "train"}),
    "popsign-test": ("extract_popsign.py", POPSIGN_SPLITS["test"], {"PER_PAIR": 4, "SPLIT": "test"}),
    "popsign-add4": ("extract_popsign.py", POPSIGN_DATASETS, {"PER_PAIR": 4, "BASE_PER_PAIR": 4, "SPLIT": "train"}),
    "aslcitizen": ("extract_aslcitizen.py", [ASLC_DATASET], {"ALL": False}),
    "wlasl": ("extract_wlasl.py", [WLASL_DATASET], {}),
}

CELL_SETUP = """
import subprocess, os, glob, json, time
import pandas as pd, numpy as np
subprocess.run(f"git clone -q {REPO_URL} /tmp/repo && git -C /tmp/repo checkout -q {REF}", shell=True, check=True)
"""

# CELL_SELECT defines `mine` for SHARD / SHARDS; LIMIT = 0 as in a full extraction. The path is
# computed exactly as the extraction cell does (train_landmark_files/<video stem>.parquet).
CELL_LOOP = """
parts = []
for SHARD in range(SHARDS):
    exec(SELECT, globals())
    parts.append(mine.assign(shard=SHARD, path=[f"train_landmark_files/{os.path.splitext(v)[0]}.parquet" for v in mine["video"]]))
inv = pd.concat(parts, ignore_index=True)
assert not inv["path"].duplicated().any(), "a video selected by two shards"
os.makedirs("/kaggle/working", exist_ok=True)
inv[["shard", "path", "video", "participant", "split", "gloss"]].to_csv("/kaggle/working/inventory.csv", index=False)
json.dump({"extractor": EXTRACTOR, "ref": REF, "cells_digest": DIGEST, "shards": SHARDS, "selection_globals": SEL,
           "selected": len(inv), "per_shard": inv.groupby("shard").size().to_dict()},
          open("/kaggle/working/inventory.json", "w"), indent=1, default=str)
print(f"{len(inv)} videos selected over {SHARDS} shards: {inv.groupby('shard').size().to_dict()}")

# frame rate and size of the selected videos (the shards do not record fps): a 300-video sample, or
# every video with --probe-all (PROBE = 0); header reads only, nothing decoded
import cv2
from concurrent.futures import ThreadPoolExecutor
def _probe(row):
    cap = cv2.VideoCapture(row[1])
    out = {"path": row[0], "fps": cap.get(cv2.CAP_PROP_FPS), "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT),
           "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    cap.release()
    return out
picked = inv if PROBE == 0 else inv.sample(min(PROBE, len(inv)), random_state=0)
with ThreadPoolExecutor(16) as ex:
    probe = pd.DataFrame(list(ex.map(_probe, zip(picked["path"], picked["src"]))))
print(f"probed {len(probe)} videos ({int((probe['fps'] <= 0).sum())} without a frame rate)")
probe.to_csv("/kaggle/working/video_probe.csv", index=False)
print("fps:", probe["fps"].round(2).value_counts().head(8).to_dict(), "| sizes:",
      (probe["width"].astype(int).astype(str) + "x" + probe["height"].astype(int).astype(str)).value_counts().head(5).to_dict())
"""


def notebook(extractor: str, ref: str, shards: int, probe: int = 300) -> dict:
    script, _, sel = EXTRACTORS[extractor]
    select = cells_at(ref, script, ("CELL_SELECT",))["CELL_SELECT"]

    def code(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.strip("\n").splitlines(True)}

    head = (f"REF, SHARDS, LIMIT, REPO_URL, EXTRACTOR, PROBE = {ref!r}, {shards}, 0, {REPO_URL!r}, {extractor!r}, {probe}\n"
            f"SEL = {sel!r}\nglobals().update(SEL)\nSELECT = {select!r}\nDIGEST = {cells_digest(select)!r}\n")
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [f"Selection inventory of `{script}` at {ref[:7]} ({shards} shards). "
                        "Generated by `scripts/extraction_inventory.py`; nothing is extracted."]},
            code(head + CELL_SETUP), code(CELL_LOOP),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["push", "status"])
    p.add_argument("extractor", choices=sorted(EXTRACTORS))
    p.add_argument("--ref", help="the commit the extraction notebooks were pinned to (required for push)")
    p.add_argument("--shards", type=int)
    p.add_argument("--probe-all", action="store_true", help="probe fps / size of every selected video (default: 300)")
    p.add_argument("--name", help="notebook name (default inventory-<extractor>)")
    a = p.parse_args()
    name = a.name or f"inventory-{a.extractor}"
    if a.action == "status":
        print(kaggle("kernels", "status", f"{user()}/{name}", check=False).strip())
        return
    if not (a.ref and a.shards):
        p.error("push needs --ref (the extraction's pinned commit) and --shards")
    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", a.ref], capture_output=True, text=True,
                         check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / f"{name}.ipynb").write_text(json.dumps(notebook(a.extractor, ref, a.shards, 0 if a.probe_all else 300), indent=1))
        meta = {"id": f"{user()}/{name}", "title": name, "code_file": f"{name}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "keywords": [],
                "dataset_sources": EXTRACTORS[a.extractor][1], "kernel_sources": [], "competition_sources": [],
                "model_sources": []}
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(kaggle("kernels", "push", "-p", tmp).strip())


if __name__ == "__main__":
    main()
