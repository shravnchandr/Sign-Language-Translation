"""Same clips, two preparations: the builders' per-clip shoulder correction vs the browser's frame shape.

Landscape clips (WLASL, ASL Citizen) are stored with x and z scaled by `aspect_factor()` (per clip,
so shoulder drop / width = 0.40). The browser cannot measure that per clip: it crops the camera to
3:4 or scales x and z by (W / H) / 0.75, which for nose-relative coordinates is the same thing. The
two factors normalise different things (body proportion vs frame shape), so a gap between them is
not by itself a bug. This prepares each raw clip both ways (and uncorrected), then asks:

  geometry  shoulder drop / width, hand size / shoulder width and the 15 joint-angle cosines per hand
            — does the browser version stay inside what the training data covers (ASL Signs, portrait
            and never corrected: its p1–p99 range), and how far does it move from the builder version?
  models    single-sign exact decode of each --checkpoint on both versions (McNemar). Score held-out
            signers only (WLASL --signers test, ASL Citizen --split test): a model trained on
            builder-corrected clips of the same signers would favour the builder version.

  python -m islr.evaluation.aspect_compare --input-root {INPUT} --keyword wlasl --signers test \
    --asl-signs-root {COMP_asl_signs} --checkpoint {CKPT_DIR}/phrases01/best_phrases.pth --out {OUT}
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import config as C
from ..core.data import contract as K
from ..core.data.vocab import load_base
from ..core.data.audit_raw import shard_clips
from ..core.data.build_wlasl_lmdb import _SIGNER_OFFSET, aspect_factor, signer_split
from .cross_dataset import mcnemar

DATA = Path(__file__).parents[1] / "core" / "data"
# joint-angle triples (parent, joint, child) at the MCP / PIP / DIP of each finger, as the model's geometry
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
ANGLES = [(([0] + f)[i], ([0] + f)[i + 1], ([0] + f)[i + 2]) for f in FINGERS for i in range(3)]


def scaled(coords: np.ndarray, k: float) -> np.ndarray:
    c = coords.copy()
    c[:, 0::3] *= k
    c[:, 2::3] *= k
    return c


def geometry(coords: np.ndarray, presence: np.ndarray) -> dict:
    """Clip-level geometry: shoulder ratio (as aspect_factor measures it), dominant-hand size
    (wrist → middle MCP, xy) / shoulder width, and the mean joint-angle cosines of each present hand."""
    l, r = C.POSE_START + 33, C.POSE_START + 36
    sy = np.median((coords[:, l + 1] + coords[:, r + 1]) / 2)
    sw = np.hypot(np.median(np.abs(coords[:, l] - coords[:, r])), np.median(coords[:, l + 1] - coords[:, r + 1]))
    out = {"ratio": float(sy / max(sw, 1e-6))}
    sizes, cos = [], []
    for slot, start in ((0, C.LH_START), (1, C.RH_START)):
        p = presence[:, slot] > 0
        if p.sum() < 2:
            continue
        h = coords[p, start:start + 63].reshape(-1, 21, 3)
        sizes.append((p.sum(), float(np.median(np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=-1)) / max(sw, 1e-6))))
        a = h[:, [i for i, _, _ in ANGLES]] - h[:, [j for _, j, _ in ANGLES]]
        b = h[:, [k for _, _, k in ANGLES]] - h[:, [j for _, j, _ in ANGLES]]
        c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
        cos.append(c.mean(0))
    if sizes:
        out["hand_size"] = max(sizes)[1]  # the hand seen in more frames
        out["cos"] = np.mean(cos, 0)
    return out


def _clip(job: tuple):
    """(path, full path, frame aspect) → per-version (coords, presence, geometry) and the two factors."""
    path, full, aspect = job
    try:
        clip = K.prepare(pd.read_parquet(full), aspect)
    except Exception as e:  # noqa: BLE001 — reported
        return path, None, str(e)
    coords, pres = clip["coords"], clip["presence"]
    if not pres.any():
        return path, None, "no hand"
    k_b, _ = aspect_factor(coords)
    k_g = aspect / 0.75
    versions = {name: scaled(coords, k) for name, k in (("builder", k_b), ("browser", k_g), ("none", 1.0))}
    return path, {"k_builder": k_b, "k_browser": k_g, "presence": pres,
                  "coords": versions, "geo": {n: geometry(c, pres) for n, c in versions.items()}}, None


def reference(root: Path, n: int, workers: int) -> pd.DataFrame:
    """Geometry of n random ASL Signs clips (portrait, never corrected) — what training covers."""
    df = pd.read_csv(root / "train.csv")
    df = df.sample(min(n, len(df)), random_state=0)
    jobs = [(p, str(root / p), 0.75) for p in df["path"]]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        res = [r for _, r, e in pool.map(_clip, jobs, chunksize=16) if e is None]
    return pd.DataFrame([{**{k: v for k, v in r["geo"]["none"].items() if k != "cos"},
                          **{f"cos{i}": c for i, c in enumerate(r["geo"]["none"].get("cos", []))}} for r in res])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", required=True, type=Path, help="extraction shard folders")
    p.add_argument("--keyword", required=True, choices=["wlasl", "aslcitizen"])
    p.add_argument("--signers", choices=["all", "test"], default="test", help="WLASL: frozen signer half")
    p.add_argument("--split", default="test", help="ASL Citizen official split")
    p.add_argument("--asl-signs-root", required=True, type=Path, help="competition folder (reference geometry)")
    p.add_argument("--n-ref", type=int, default=3000)
    p.add_argument("--checkpoint", action="append", default=[], type=Path)
    p.add_argument("--sign-map", type=Path, help="default: the competition's sign_to_prediction_index_map.json")
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--limit", type=int, default=0, help="random sample of the scored clips (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sign2idx = load_base(a.sign_map or a.asl_signs_root / "sign_to_prediction_index_map.json")
    idx2sign = {v: k for k, v in sign2idx.items()}
    df = shard_clips(a.input_root, a.keyword)
    if a.keyword == "wlasl":
        gmap = json.loads((DATA / "wlasl_gloss_map.json").read_text())
        df["sign"] = [idx2sign.get(gmap.get(str(g).lower())) for g in df["gloss"]]
        if a.signers == "test":
            pid = pd.to_numeric(df["participant"]) + _SIGNER_OFFSET
            df = df[pid.isin(signer_split(pd.DataFrame({"participant_id": pid}))["test"])]
    else:
        gmap = json.loads((DATA / "aslcitizen_gloss_map.json").read_text())
        df["sign"] = [gmap[g]["sign"] if isinstance(gmap.get(g), dict) and gmap[g]["kind"] == "identical" else None
                      for g in df["gloss"]]
        df = df[df["split"] == a.split]
    df = df[df["sign"].notna()]
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)
    print(f"{len(df)} {a.keyword} clips (signers {a.signers if a.keyword == 'wlasl' else 'split ' + a.split})")

    ref = reference(a.asl_signs_root, a.n_ref, a.workers)
    ref.to_csv(a.out / "reference_aslsigns.csv", index=False)

    models = []
    for ck in a.checkpoint:
        from ..export.export_onnx import load_ctc_model

        meta = json.loads(ck.with_name("phrases_meta.json").read_text())
        models.append((ck.parent.name, load_ctc_model(ck, meta, len(sign2idx))))
    blank = len(sign2idx)

    @torch.no_grad()
    def exact(model, coords, presence, y) -> bool:
        x = K.clip_input(torch.from_numpy(coords), torch.from_numpy(presence), a.max_frames, trim=True)
        out = model(x[None].float(), torch.ones(1, len(x), dtype=torch.bool))[0]
        hyp, prev = [], None
        for c in out.argmax(-1).tolist():
            if c != prev and c != blank:
                hyp.append(c)
            prev = c
        return hyp == [y]

    rows, errors = [], []
    jobs = list(zip(df["path"], df["full_path"], df["width"] / df["height"]))
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        for r, (path, res, err) in zip(df.itertuples(), pool.map(_clip, jobs, chunksize=8)):
            if err:
                errors.append((path, err))
                continue
            row = {"path": path, "sign": r.sign, "signer": r.participant, "width": r.width, "height": r.height,
                   "k_builder": res["k_builder"], "k_browser": res["k_browser"]}
            for v, g in res["geo"].items():
                row.update({f"{v}_{m}": g.get(m, np.nan) for m in ("ratio", "hand_size")})
            gb, gg = res["geo"]["builder"], res["geo"]["browser"]
            if "cos" in gb:
                row["cos_builder_vs_browser"] = float(np.abs(gb["cos"] - gg["cos"]).mean())
                for v in ("builder", "browser"):
                    for i, c in enumerate(res["geo"][v]["cos"]):
                        row[f"{v}_cos{i}"] = float(c)
            for name, m in models:
                for v in ("builder", "browser"):
                    row[f"{name}_{v}"] = exact(m, res["coords"][v], res["presence"], sign2idx[r.sign])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(a.out / f"aspect_compare_{a.keyword}.csv", index=False)

    def outside(col_ref: str, col: str) -> float:
        lo, hi = ref[col_ref].quantile([0.01, 0.99])
        return float(((res[col] < lo) | (res[col] > hi)).mean())

    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    cos_cols = [f"cos{i}" for i in range(len(ANGLES))]
    ref_cos_lo, ref_cos_hi = ref[cos_cols].quantile(0.01).to_numpy(), ref[cos_cols].quantile(0.99).to_numpy()

    def cos_outside(v: str) -> float:
        c = res[[f"{v}_{x}" for x in cos_cols]].dropna().to_numpy()
        return float(((c < ref_cos_lo) | (c > ref_cos_hi)).mean()) if len(c) else float("nan")  # 0.02 if like ASL Signs

    summary = {
        "dataset": a.keyword, "clips": len(res), "errors": len(errors), "error_examples": errors[:5],
        "k": {"builder": res["k_builder"].quantile(q).round(3).tolist(),
              "browser": res["k_browser"].quantile(q).round(3).tolist(),
              "builder / browser": (res["k_builder"] / res["k_browser"]).quantile(q).round(3).tolist(),
              "builder fell back to 1": int((res["k_builder"] == 1.0).sum())},
        "quantiles": q,
        "ratio": {"aslsigns_reference": ref["ratio"].quantile(q).round(3).tolist(),
                  **{v: res[f"{v}_ratio"].quantile(q).round(3).tolist() for v in ("builder", "browser", "none")},
                  "outside_reference_p1_p99": {v: outside("ratio", f"{v}_ratio") for v in ("builder", "browser", "none")}},
        "hand_size": {"aslsigns_reference": ref["hand_size"].quantile(q).round(3).tolist(),
                      **{v: res[f"{v}_hand_size"].quantile(q).round(3).tolist() for v in ("builder", "browser", "none")},
                      "outside_reference_p1_p99": {v: outside("hand_size", f"{v}_hand_size")
                                                   for v in ("builder", "browser", "none")}},
        "joint_cosines": {"mean_abs_change_builder_vs_browser": res["cos_builder_vs_browser"].quantile(q).round(4).tolist(),
                          "angle_values_outside_reference_p1_p99 (ASL Signs itself: 0.02)": {v: cos_outside(v) for v in ("builder", "browser")},
                          "reference_between_clip_spread (mean abs dev)": float((ref[cos_cols] - ref[cos_cols].median()).abs().mean().mean())},
        "models": {name: mcnemar(res[f"{name}_builder"], res[f"{name}_browser"]) for name, _ in models},
    }
    (a.out / f"aspect_compare_{a.keyword}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
