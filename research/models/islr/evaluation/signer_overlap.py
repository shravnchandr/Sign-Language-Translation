"""Do signers of one dataset reappear in another? Body-proportion fingerprints, self-calibrated.

PopSign ASL comes from the same game as Google ASL Signs, so some signers may be in both — then
ASL Signs held-out scores of a model trained on PopSign are not clean. Neither dataset links its
anonymised ids, so this compares what landmarks keep of a person: hand bone proportions (20 bones /
palm length, 3D, detected frames only, both hands pooled) and face / body spacing (pose eye, ear and
mouth width, shoulder drop, lip width, inner-eyebrow gap, eyebrow-to-lip — each / shoulder width).
Per clip: median over frames; per signer: median over clips; features standardised by the spread
between the reference signers and weighted by their split-half reliability r (weight r / (1 − r):
the face / body spacings, r ≈ 0.99 on ASL Signs, outweigh the noisier hand bones, r 0.6–0.9).

Calibration: each reference signer's clips are split in two random halves. Same-person distances
(half A of i vs half B of i) and different-person distances (A of i vs B of j) give the scale; the
AUC says how well the fingerprint tells people apart. Each query signer's nearest reference signer
is then reported with the share of different-person pairs that are closer (≈ 0 and inside the
same-person range = likely the same person) — a screen, not proof. The calibration measures
consistency *within* one dataset; matching a person across datasets (other phone, session,
framing) is harder and unvalidated, so "not flagged" does not establish that a signer is absent.

Usage:
  PYTHONPATH=research/models python -m islr.evaluation.signer_overlap \
    --ref-data-dir data/asl-is-lmdb --ref-lmdb-path data/asl-is-lmdb/is.lmdb.mdb \
    --query-data-dir /tmp/popsign --query-lmdb-path /tmp/popsign/popsign.lmdb.mdb --out overlap.json
"""

import argparse
import io
import json
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch

from ..core import config as C
from ..core.data._cache_keys import lmdb_key
from ..core.data.preprocessing import hand_presence

_L = C.COORD_FEAT // 3
_POSE, _RH, _FACE = C.POSE_START // 3, C.RH_START // 3, C.FACE_START // 3
_F = {i: _FACE + k for k, i in enumerate(C.SELECTED_FACE_INDICES)}
_BONES = [(C.HAND_PARENT[c], c) for c in range(1, C.N_LH) if c != 9]  # 0→9 is the palm unit
_BODY = {  # name: (a, b) landmark pairs, length / shoulder width
    "eye_width": (_POSE + 2, _POSE + 5), "ear_width": (_POSE + 7, _POSE + 8),
    "pose_mouth_width": (_POSE + 9, _POSE + 10), "lip_width": (_F[61], _F[291]),
    "inner_brow_gap": (_F[107], _F[336]), "brow_to_lip": (_F[105], _F[0]),
}
FEATURES = [f"bone_{p}_{c}" for p, c in _BONES] + list(_BODY) + ["shoulder_drop"]


def clip_features(coords: torch.Tensor, aspect: float) -> np.ndarray:
    """(T, COORD_FEAT) stored coords → per-clip fingerprint (NaN where nothing was detected)."""
    filled, pres = hand_presence(coords)
    x = filled.numpy().reshape(len(filled), _L, 3).copy()
    x[..., 1] /= aspect  # y is a fraction of height, x of width: put both in width units
    out = np.full(len(FEATURES), np.nan)
    hands = []
    for slot, start in ((0, 0), (1, _RH)):
        h = x[pres[:, slot].numpy() > 0, start:start + C.N_LH]
        if len(h):
            palm = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)
            ok = palm > 1e-3
            bones = np.stack([np.linalg.norm(h[:, c] - h[:, p], axis=-1) for p, c in _BONES], 1)
            if ok.any():
                hands.append(np.median(bones[ok] / palm[ok, None], 0))
    if hands:
        out[:len(_BONES)] = np.mean(hands, 0)
    sh = np.linalg.norm(x[:, _POSE + 11, :2] - x[:, _POSE + 12, :2], axis=-1)
    ok = sh > 1e-3
    if ok.any():
        for k, (a, b) in enumerate(_BODY.values()):
            out[len(_BONES) + k] = np.median(np.linalg.norm(x[ok, a, :2] - x[ok, b, :2], axis=-1) / sh[ok])
        mid = (x[ok, _POSE + 11, 1] + x[ok, _POSE + 12, 1]) / 2
        out[-1] = np.median((mid - x[ok, _POSE, 1]) / sh[ok])
    return out


def load_features(data_dir: Path, lmdb_path: Path, per_signer: int, aspect: float, seed: int) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "train.csv")
    df = df.sample(frac=1, random_state=seed).groupby("participant_id").head(per_signer)
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False)
    # per-clip frame width / height when the CSV has the frame size (PopSign: 3:4 and 9:16 clips)
    asp = (df["width"] / df["height"]).fillna(aspect) if {"width", "height"} <= set(df) else pd.Series(aspect, df.index)
    rows = []
    with env.begin(buffers=True) as txn:
        for pid, path, k in zip(df["participant_id"], df["path"], asp):
            val = txn.get(lmdb_key(path))
            if val is not None:
                rows.append([pid, *clip_features(torch.load(io.BytesIO(bytes(val)), weights_only=True), k)])
    env.close()
    return pd.DataFrame(rows, columns=["signer", *FEATURES])


def signer_vectors(f: pd.DataFrame) -> pd.DataFrame:
    return f.groupby("signer")[FEATURES].median()


def distance(a: pd.DataFrame, b: pd.DataFrame, scale: pd.Series, w: pd.Series) -> pd.DataFrame:
    """Weighted RMS standardised difference over features both have, a.index × b.index."""
    A, B = (a / scale).to_numpy(), (b / scale).to_numpy()
    sq = (A[:, None] - B[None]) ** 2
    wt = np.where(np.isnan(sq), 0, w.to_numpy())
    d = (np.nansum(sq * wt, -1) / wt.sum(-1)) ** 0.5
    return pd.DataFrame(d, index=a.index, columns=b.index)


def reliability_weights(f: pd.DataFrame, seed: int) -> pd.Series:
    """Split-half correlation r of each feature's signer medians → weight r / (1 − r)."""
    A, B = halves(f, seed + 1)  # a different split than the calibration's
    r = pd.Series({c: A[c].corr(B[c]) for c in FEATURES}).clip(0, 0.995).fillna(0)
    return r / (1 - r)


def halves(f: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = np.random.default_rng(seed).random(len(f)) < 0.5
    return signer_vectors(f[r]), signer_vectors(f[~r])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref-data-dir", required=True, type=Path)
    p.add_argument("--ref-lmdb-path", required=True, type=Path)
    p.add_argument("--query-data-dir", required=True, type=Path)
    p.add_argument("--query-lmdb-path", required=True, type=Path)
    p.add_argument("--per-signer", type=int, default=200, help="clips sampled per signer")
    p.add_argument("--aspect", type=float, default=0.75,
                   help="frame width / height when train.csv has no width / height columns")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    ref = load_features(a.ref_data_dir, a.ref_lmdb_path, a.per_signer, a.aspect, a.seed)
    qry = load_features(a.query_data_dir, a.query_lmdb_path, a.per_signer, a.aspect, a.seed)
    print(f"reference {ref['signer'].nunique()} signers / {len(ref)} clips; "
          f"query {qry['signer'].nunique()} signers / {len(qry)} clips")
    scale = signer_vectors(ref).std().replace(0, np.nan)
    w = reliability_weights(ref, a.seed)
    print("feature weights (r / (1 - r)): " + ", ".join(f"{k} {v:.0f}" for k, v in w.sort_values(ascending=False).items()))

    def calibrate(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        A, B = halves(f, a.seed)
        d = distance(A, B, scale, w)
        same = np.diag(d.loc[A.index, A.index].to_numpy()) if A.index.equals(B.index) else np.array(
            [d.loc[s, s] for s in A.index.intersection(B.index)])
        diff = np.array([d.loc[i, j] for i in d.index for j in d.columns if i != j])
        return same, diff

    def auc(same: np.ndarray, diff: np.ndarray) -> float:
        return float((same[:, None] < diff[None]).mean())

    report = {"features": FEATURES, "weights": w.round(2).to_dict()}
    for name, f in (("reference", ref), ("query", qry)):
        same, diff = calibrate(f)
        report[f"{name}_calibration"] = {
            "same_person": {"median": float(np.median(same)), "max": float(same.max())},
            "different": {"min": float(diff.min()), "p05": float(np.quantile(diff, .05)),
                          "median": float(np.median(diff))},
            "auc": auc(same, diff)}
        print(f"{name} split-half: same-person median {np.median(same):.2f} max {same.max():.2f}; "
              f"different min {diff.min():.2f} p05 {np.quantile(diff, .05):.2f} median "
              f"{np.median(diff):.2f}; AUC {auc(same, diff):.3f}")
    same_ref, diff_ref = calibrate(ref)
    d = distance(signer_vectors(qry), signer_vectors(ref), scale, w)
    rows = []
    for q, r in d.iterrows():
        best = r.idxmin()
        rows.append({"query": int(q), "nearest_ref": int(best), "distance": float(r[best]),
                     "second": float(r.drop(best).min()),
                     "closer_different_pairs": float((diff_ref < r[best]).mean()),
                     "within_same_range": bool(r[best] <= same_ref.max())})
    rows.sort(key=lambda x: x["distance"])
    report["query_nearest"] = rows
    likely = [x for x in rows if x["within_same_range"] and x["closer_different_pairs"] < 0.01]
    report["likely_same_person"] = likely
    print("query signer → nearest reference signer (distance, 2nd, share of different-person pairs closer):")
    for x in rows:
        print(f"  {x['query']} → {x['nearest_ref']}  {x['distance']:.2f} ({x['second']:.2f}) "
              f"{x['closer_different_pairs']:.3f}" + ("  ← likely same" if x in likely else ""))
    print(f"{len(likely)} of {len(rows)} query signers likely also in the reference set")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
