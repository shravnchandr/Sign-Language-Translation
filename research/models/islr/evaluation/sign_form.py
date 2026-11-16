"""Form screen for a failure review: does a sign look the same, measurably, in two datasets?

Per clip (stored contract-v4 coordinates, trimmed to the hand span): the dominant hand (the slot
detected in more frames; ties → more wrist motion), with x negated when it is the right slot so both
chiralities share one side; the median shoulder width (pose 11–12, xy) as the unit; then
  loc_x, loc_y        median dominant-wrist position relative to the nose (shoulder widths)
  start_y, end_y      the same over the first / last fifth of its detected frames
  path                dominant-wrist path length (shoulder widths)
  both_share          frames with both hands detected / frames with any hand
  cos_<finger><j>     median joint-angle cosines of the dominant hand (15: 3 per finger)
Per sign and dataset: medians (`sign_form_by_sign.csv`). Per clip of the reviewed dataset: distance
to the reference dataset's centroid of its own label and of the --confusion label, in units of the
reference within-sign spread (median absolute deviation pooled over signs), overall and per feature
group (location / motion / handshape / two-hand) → `sign_form_clips.csv`. A clip nearer the confusion
than its own label in some group says where the forms differ; it does not say why (variant, mapping,
tracking or recording) — that is for a reviewer who knows the signs. Writes aggregates and derived
numbers only, no landmarks.

  PYTHONPATH=research/models python -m islr.evaluation.sign_form \
      --ref-dir {IS_LMDB_DIR} --ref-lmdb {IS_LMDB_DIR}/is.lmdb.mdb \
      --data-dir {ASLC_DIR} --lmdb-path {ASLC_DIR}/aslcitizen.lmdb.mdb \
      --sign cowboy:dog --sign hen:pen ... --out {OUT}/sign_form
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import config as C
from ..core.data._cache_keys import lmdb_key
from ..core.data.contract import load_clip, segment
from ..core.data.dataset import _open_lmdb_env
from ..core.data.validate import require_accepted

CHAINS = {"thumb": [0, 1, 2, 3, 4], "index": [0, 5, 6, 7, 8], "middle": [0, 9, 10, 11, 12],
          "ring": [0, 13, 14, 15, 16], "pinky": [0, 17, 18, 19, 20]}
GROUPS = {"location": ["loc_x", "loc_y", "start_y", "end_y"], "motion": ["path"], "two_hand": ["both_share"],
          "handshape": [f"cos_{f}{j}" for f in CHAINS for j in range(3)]}


def _lm(coords: np.ndarray, start: int, i: int) -> np.ndarray:
    return coords[:, start + 3 * i: start + 3 * i + 3]


def features(coords: np.ndarray, pres: np.ndarray) -> dict | None:
    s = segment(__import__("torch").from_numpy(pres))
    coords, pres = coords[s], pres[s] > 0
    if not pres.any():
        return None
    starts = (C.LH_START, C.RH_START)
    wrist = [_lm(coords, st, 0)[:, :2] for st in starts]
    motion = [float(np.linalg.norm(np.diff(w[p], axis=0), axis=1).sum()) if p.sum() > 1 else 0.0
              for w, p in zip(wrist, pres.T)]
    counts = pres.sum(0)
    h = 0 if (counts[0], motion[0]) >= (counts[1], motion[1]) else 1
    sign_x = 1.0 if h == 0 else -1.0
    sw = np.median(np.linalg.norm(_lm(coords, C.POSE_START, 11)[:, :2] - _lm(coords, C.POSE_START, 12)[:, :2], axis=1))
    sw = max(float(sw), 1e-3)
    det = pres[:, h]
    w = wrist[h][det] * np.array([sign_x, 1.0]) / sw  # stored coords are nose-relative
    k = max(1, len(w) // 5)
    f = {"loc_x": float(np.median(w[:, 0])), "loc_y": float(np.median(w[:, 1])),
         "start_y": float(np.median(w[:k, 1])), "end_y": float(np.median(w[-k:, 1])),
         "path": motion[h] / sw, "both_share": float(pres.all(1).sum() / max(pres.any(1).sum(), 1))}
    hand = np.stack([_lm(coords, starts[h], i) for i in range(21)], 1)[det]  # (T, 21, 3)
    for name, ch in CHAINS.items():
        for j in range(3):
            a = hand[:, ch[j + 1]] - hand[:, ch[j]]
            b = hand[:, ch[j + 2]] - hand[:, ch[j + 1]]
            cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-6)
            f[f"cos_{name}{j}"] = float(np.median(cos))
    return f


def clip_table(data_dir: Path, lmdb_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    require_accepted(data_dir, lmdb_path)
    env = _open_lmdb_env(str(lmdb_path))
    rows = []
    with env.begin() as txn:
        for r in df.itertuples():
            c, p = load_clip(txn.get(lmdb_key(r.path)))
            f = features(c.numpy(), p.numpy())
            if f is not None:
                rows.append({"path": r.path, "sign": r.sign, "signer": r.participant_id,
                             "split": getattr(r, "split", None), **f})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref-dir", required=True, type=Path)
    p.add_argument("--ref-lmdb", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--lmdb-path", required=True, type=Path)
    p.add_argument("--sign", action="append", required=True, help="label:confusion (ASL Signs label names)")
    p.add_argument("--ref-per-sign", type=int, default=300, help="reference clips per sign (all signers, seeded)")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    pairs = [tuple(s.split(":")) for s in a.sign]
    signs = sorted({x for pr in pairs for x in pr})
    ref = pd.read_csv(a.ref_dir / "train.csv")
    ref = ref[ref["sign"].isin(signs)]
    ref = ref.sample(frac=1.0, random_state=0).groupby("sign").head(a.ref_per_sign)
    tgt = pd.read_csv(a.data_dir / "train.csv")
    tgt = tgt[tgt["sign"].isin(signs)]
    R, T = clip_table(a.ref_dir, a.ref_lmdb, ref), clip_table(a.data_dir, a.lmdb_path, tgt)
    feats = [c for g in GROUPS.values() for c in g]
    a.out.mkdir(parents=True, exist_ok=True)
    by = pd.concat([R.assign(dataset="reference"), T.assign(dataset="reviewed")])
    agg = by.groupby(["sign", "dataset"])[feats].median()
    agg.insert(0, "clips", by.groupby(["sign", "dataset"]).size())
    agg.round(4).to_csv(a.out / "sign_form_by_sign.csv")
    cent = R.groupby("sign")[feats].median()
    mad = R.groupby("sign")[feats].apply(lambda g: (g - g.median()).abs().median()).median()
    scale = mad.where(mad > 1e-3, R[feats].std()).replace(0, 1.0)
    out = []
    for lab, conf in pairs:
        for r in T[T["sign"] == lab].itertuples():
            v = pd.Series({f: getattr(r, f) for f in feats})
            row = {"path": r.path, "sign": lab, "confusion": conf, "signer": r.signer, "split": r.split}
            for g, cols in list(GROUPS.items()) + [("all", feats)]:
                dl = float(np.sqrt((((v[cols] - cent.loc[lab, cols]) / scale[cols]) ** 2).mean()))
                dc = float(np.sqrt((((v[cols] - cent.loc[conf, cols]) / scale[cols]) ** 2).mean()))
                row[f"d_{g}_label"], row[f"d_{g}_conf"] = dl, dc
                row[f"{g}_nearer_conf"] = dc < dl
            out.append(row)
    clips = pd.DataFrame(out)
    clips.round(4).to_csv(a.out / "sign_form_clips.csv", index=False)
    # spread of the reference itself: its own clips' distance to their centroid (what "near" means)
    own = {}
    for lab, _ in pairs:
        g = R[R["sign"] == lab]
        own[lab] = float(np.median(np.sqrt((((g[feats] - cent.loc[lab]) / scale) ** 2).mean(1))))
    summ = clips.groupby(["sign", "confusion"]).agg(
        clips=("path", "size"),
        **{f"{g}_nearer_conf": (f"{g}_nearer_conf", "mean") for g in list(GROUPS) + ["all"]},
        d_all_label=("d_all_label", "median"), d_all_conf=("d_all_conf", "median")).reset_index()
    summ["ref_own_d_all"] = summ["sign"].map(own)
    summ.round(3).to_csv(a.out / "sign_form_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string(index=False))
    print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
