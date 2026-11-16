"""Who is in a PopSign ASL v1.0 split? Lists the videos of a mounted split by file name only (no video
is opened): per participant, videos and signs; the user number is the digits after the last "." of the
participant prefix (`4a.6001` → 6001, `gtsignstudy4a.8031` → 8031), as `build_popsign_lmdb` offsets it
(participant_id = 700000 + number). With --train-csv (an accepted PopSign release's train.csv), reports
which participants also supply training clips. Participant matching is by number only; it says nothing
about overlap with ASL Signs signers (other ids).

  PYTHONPATH=research/models python -m islr.core.data.popsign_signers --root {DATASET_popsign_asl_v1_0_game_test} \
      --train-csv {POPSIGN_DIR}/train.csv --out {OUT}
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def list_videos(root: Path) -> pd.DataFrame:
    """The extraction's own parse (scripts/extract_popsign.py CELL_SELECT): sign = folder name."""
    rows = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".mp4"):
                sign = os.path.basename(dirpath)
                cut = f.find(f"-{sign}-")
                rows.append({"video": f, "gloss": sign, "participant": f[:cut] if cut > 0 else f.split("-")[0]})
    return pd.DataFrame(rows)


def user_number(participant: str) -> int | None:
    m = re.search(r"\.(\d+)$", participant)
    return int(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--train-csv", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    v = list_videos(a.root)
    if v.empty:
        raise SystemExit(f"no .mp4 under {a.root}")
    v["user"] = v["participant"].map(user_number)
    bad = v[v["user"].isna()]["participant"].unique()
    if len(bad):
        raise SystemExit(f"participant prefixes without a user number: {list(bad)[:5]}")
    per = v.groupby(["participant", "user"]).agg(videos=("video", "size"), signs=("gloss", "nunique")).reset_index()
    if a.train_csv:
        t = pd.read_csv(a.train_csv)
        train = set((t["participant_id"] - 700000).astype(int))
        per["in_training"] = per["user"].isin(train)
        print(f"training release: {len(train)} PopSign users")
    a.out.mkdir(parents=True, exist_ok=True)
    per.to_csv(a.out / "popsign_split_signers.csv", index=False)
    v.groupby("gloss").size().rename("videos").to_csv(a.out / "popsign_split_signs.csv")
    print(f"{len(v):,} videos, {len(per)} participants, {v['gloss'].nunique()} signs")
    print(per.to_string(index=False))
    if "in_training" in per:
        both = per[per["in_training"]]
        print(f"participants also in training: {len(both)} ({int(both['videos'].sum()):,} videos)")


if __name__ == "__main__":
    main()
