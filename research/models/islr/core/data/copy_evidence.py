"""Per-frame copy-rule evidence (ghost_review.frame_evidence) for chosen raw parquets — for datasets
without videos or extraction shards (ASL Signs: competition parquets only), where the rule's transfer
can be checked only statistically against the visually labelled WLASL / PopSign frames.

  PYTHONPATH=research/models python -m islr.core.data.copy_evidence --root {COMP_asl_signs} \
      --clips <audit_aslsigns_clips.csv> --min-copies 1 --aspect 0.75 --out {OUT}

--clips: a CSV with `path` (relative to --root); with `ghost_frames`, --min-copies keeps clips with at
least that many v4 copy frames. Writes copy_evidence.csv: one row per frame with both hand slots
detected (frame, path, iou, wrist distances, slot_landmark_dist, statuses, assignment_uncertain).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def clip_evidence(job: tuple) -> pd.DataFrame:
    from islr.core.data import contract as K
    from islr.core.data.ghost_review import frame_evidence

    root, path, aspect = job
    raw = K.raw_arrays(pd.read_parquet(Path(root) / path))
    posed = ~np.isnan(raw["pose"]).all((1, 2))
    raw = {k: v[posed] for k, v in raw.items()}
    status, uncertain, _ = K.resolve_ghosts(raw, aspect)
    ev = frame_evidence(raw, status, aspect)
    ev["assignment_uncertain"] = uncertain[ev["frame"].to_numpy()].astype(int) if len(ev) else []
    ev["path"] = path
    return ev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--clips", required=True, type=Path)
    p.add_argument("--min-copies", type=int, default=0)
    p.add_argument("--aspect", type=float, default=0.75)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    c = pd.read_csv(a.clips)
    if a.min_copies:
        c = c[c["ghost_frames"] >= a.min_copies]
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(clip_evidence, [(str(a.root), x, a.aspect) for x in c["path"]], chunksize=8))
    ev = pd.concat(parts, ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out / "copy_evidence.csv", index=False)
    copy = (ev["left_status"] == "GHOST") | (ev["right_status"] == "GHOST")
    print(f"{len(c)} clips, {len(ev)} two-slot frames, {int(copy.sum())} copy decisions, "
          f"{int(ev['assignment_uncertain'].sum())} assignment-uncertain")


if __name__ == "__main__":
    main()
