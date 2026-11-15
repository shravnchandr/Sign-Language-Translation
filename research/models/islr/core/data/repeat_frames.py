"""Repeated-frame screen from landmarks: was a clip recorded as ~30 fps frames each repeated k times?

PopSign's 120 fps videos are 30 fps frames repeated 4× (60 fps: 2×; video_timing). MediaPipe smooths its
landmarks over time, so repeated images do not give equal coordinates — but they give a periodic motion
pattern: the step onto a new image is large, the steps between repeats small. Per clip, from the raw
frames (pose-less frames dropped): m_t = mean displacement of the upper-body pose landmarks (0–24, x·aspect
and y) between frames t and t + 1, divided by the clip's median shoulder width; for k in (2, 4), the
phase share s_k = max over φ of Σ m[φ::k] / Σ m (1/k for motion spread evenly over the phases, → 1 when
all motion falls on one phase). Held poses lower every phase alike, so they do not raise s_k. Also the
median s_4 over 24-frame windows (robust to a phase slip), the mean motion and the clip length.

Labels for calibration come from elsewhere (PopSign's probed fps); this module only measures.

  PYTHONPATH=research/models python -m islr.core.data.repeat_frames --input-root {INPUT} --keyword popsign \
      --aslsigns-root {COMP_asl_signs} --save-seq <paths.csv> --out {OUT}
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

UPPER = np.arange(25)  # pose landmarks above the hips (face, shoulders, arms, hands' pose points)
WINDOW = 24


def phase_share(m: np.ndarray, k: int) -> float:
    n = len(m) // k * k
    if n < 2 * k or m[:n].sum() <= 0:
        return np.nan
    e = m[:n].reshape(-1, k).sum(0)
    return float(e.max() / e.sum())


def motion(df: pd.DataFrame, aspect: float) -> np.ndarray:
    from islr.core.data import contract as K

    raw = K.raw_arrays(df)
    pose = raw["pose"][~np.isnan(raw["pose"]).all((1, 2))][:, :, :2] * np.array([aspect, 1.0])
    if len(pose) < 3:
        return np.zeros(0)
    sw = np.nanmedian(np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1))
    return np.nanmean(np.linalg.norm(np.diff(pose[:, UPPER], axis=0), axis=-1), axis=1) / max(sw, 1e-6)


def clip_features(job: tuple) -> tuple[dict, np.ndarray | None]:
    path, full, aspect, keep_seq = job
    try:
        m = motion(pd.read_parquet(full), aspect)
    except Exception as e:  # noqa: BLE001 — reported per clip
        return {"path": path, "error": str(e)}, None
    win = [phase_share(m[i:i + WINDOW], 4) for i in range(0, max(len(m) - WINDOW + 1, 1), WINDOW // 2)]
    f = {"path": path, "T": len(m) + 1, "mean_motion": float(m.mean()) if len(m) else np.nan,
         "s2": phase_share(m, 2), "s4": phase_share(m, 4),
         "s4_window_median": float(np.nanmedian(win)) if np.isfinite(win).any() else np.nan, "error": None}
    return f, (m if keep_seq else None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", type=Path, help="extraction shard folders (shard_*.csv)")
    p.add_argument("--keyword", default="popsign")
    p.add_argument("--aslsigns-root", type=Path, help="competition folder (train.csv + parquets)")
    p.add_argument("--save-seq", type=Path, help="CSV with `path`: keep these clips' motion sequences")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    keep = set(pd.read_csv(a.save_seq)["path"]) if a.save_seq else set()
    jobs = []
    if a.input_root:
        from islr.core.data.audit_raw import shard_clips

        c = shard_clips(a.input_root, a.keyword)
        asp = (c["width"] / c["height"]).fillna(0.75) if "width" in c else pd.Series(0.75, c.index)
        jobs += [(f"{a.keyword}:{x}", f, float(r), x in keep) for x, f, r in zip(c["path"], c["full_path"], asp)]
    if a.aslsigns_root:
        t = pd.read_csv(a.aslsigns_root / "train.csv")
        jobs += [(f"aslsigns:{x}", str(a.aslsigns_root / x), 0.75, x in keep) for x in t["path"]]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, seqs = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for f, m in ex.map(clip_features, jobs, chunksize=32):
            rows.append(f)
            if m is not None:
                seqs += [{"path": f["path"], "t": i, "m": float(v)} for i, v in enumerate(m)]
    res = pd.DataFrame(rows)
    res["dataset"], res["path"] = res["path"].str.split(":", n=1).str[0], res["path"].str.split(":", n=1).str[1]
    res.to_csv(a.out / "repeat_frames.csv", index=False)
    pd.DataFrame(seqs).to_csv(a.out / "repeat_frames_seq.csv", index=False)
    print(res.groupby("dataset")[["T", "s2", "s4", "s4_window_median"]].describe().T.round(3).to_string())
    print(f"errors: {int(res['error'].notna().sum())}")


if __name__ == "__main__":
    main()
