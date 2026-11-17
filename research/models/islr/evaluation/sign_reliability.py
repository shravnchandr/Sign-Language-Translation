"""Per-sign reliability of a phrase (CTC) checkpoint, for the app: how often each sign is
recognised by signers the model never saw, what it is confused with, and how far a score
can be trusted.

Clips: ASL Signs held-out fold (--val-fold, the signers the checkpoint did not train on)
and, with --wlasl-dir, WLASL's held-out signer half (build_wlasl_lmdb.signer_split "test").
Each clip is preprocessed as in the browser (hand-span trim, ≤ --max-frames) and scored
"pooled": per sign, the highest per-frame probability over the segment, blank excluded;
signs ranked by it. The JSON states this so the app can compute the same ranking.

Output JSON:
  overall      top-1 / top-3 / top-5 and exact single-sign decode, per source and combined
  calibration  top-1 accuracy by top-1 pooled probability, and the thresholds that give
               80 / 90 / 95 % top-1 precision with the share of clips they keep
  emitted      the app's "not sure" calibration: every sign the greedy CTC decode emits, scored as
               ASL-App's decodeWithScores does (pooled over the sign's own stretch, stretches split
               halfway between neighbouring emissions), correct iff it is the clip's label; same
               table and thresholds. Single-sign clips only — not validated on multi-sign phrases
Thresholds: score groups are never split (ties are kept or dropped together), the cutoff is the exact
score (unrounded; keep iff score >= cutoff), and its precision is re-measured by applying it.
  signs        per sign: n, top1, top3, top5, exact, confused_with [[sign, share of clips]]

Usage:
  PYTHONPATH=research/models uv run python -m islr.evaluation.sign_reliability \
    --checkpoint <best_phrases.pth> --data-dir data/asl-is-lmdb \
    --lmdb-path data/asl-is-lmdb/is.lmdb.mdb --wlasl-dir data/wlasl-lmdb --out exports/phrases04/sign_reliability.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.data._cache_keys import lmdb_key
from ..core.data.build_wlasl_lmdb import signer_split
from ..core.data.contract import clip_input, load_clip
from ..core.data.vocab import load_base
from ..core.data.dataset import _open_lmdb_env, signer_fold_split
from ..core.data.validate import require_accepted
from ..export.export_onnx import load_ctc_model


def _env(path: Path):
    return _open_lmdb_env(str(path))  # the handle require_accepted opened (one env per process)


def score_clips(model, df: pd.DataFrame, lmdb_path: Path, sign2idx: dict, max_frames: int,
                source: str) -> pd.DataFrame:
    blank = len(sign2idx)
    rows = []
    with _env(lmdb_path).begin() as txn, torch.no_grad():
        for r in df.itertuples():
            x = clip_input(*load_clip(txn.get(lmdb_key(r.path))), max_frames, trim=True)
            out = model(x[None].float(), torch.ones(1, len(x), dtype=torch.bool))[0]
            hyp, prev = [], None
            for c in out.argmax(-1).tolist():
                if c != prev and c != blank:
                    hyp.append(c)
                prev = c
            probs = out.float().softmax(-1)
            pooled = probs[:, :blank].max(0).values
            top = pooled.topk(5)
            y = sign2idx[r.sign]
            emitted = emitted_scores(out.float(), blank)
            assert [c for c, _ in emitted] == hyp, (r.path, emitted, hyp)
            rows.append(dict(source=source, signer=r.participant_id, sign=y,
                             ranked=top.indices.tolist(), p1=float(top.values[0]),
                             exact=hyp == [y], emitted=[c for c, _ in emitted],
                             emitted_scores=[v for _, v in emitted]))
    res = pd.DataFrame(rows)
    for k in (1, 3, 5):
        res[f"top{k}"] = [y in t[:k] for y, t in zip(res["sign"], res["ranked"])]
    res["pred"] = [t[0] for t in res["ranked"]]
    return res


def emitted_scores(logits: torch.Tensor, blank: int) -> list[tuple[int, float]]:
    """ASL-App decodeWithScores on (T, C) logits: greedy CTC runs from the logits' argmax (as the
    app and the decode do — float32 softmax can round distinct logits into a tie); each emitted sign
    pooled (max over frames of its softmax probability) over its own stretch — from halfway after
    the previous run to halfway before the next (floor((end_prev + start) / 2) + 1 ..
    floor((end + start_next) / 2))."""
    probs = logits.softmax(-1)
    runs, prev = [], -1
    for t, c in enumerate(logits.argmax(-1).tolist()):
        if c != prev and c != blank:
            runs.append([c, t, t])
        elif c == prev and c != blank:
            runs[-1][2] = t
        prev = c
    out = []
    for i, (c, start, end) in enumerate(runs):
        lo = 0 if i == 0 else (runs[i - 1][2] + start) // 2 + 1
        hi = len(probs) - 1 if i == len(runs) - 1 else (end + runs[i + 1][1]) // 2
        out.append((c, float(probs[lo:hi + 1, c].max())))
    return out


def cutoffs(score: np.ndarray, correct: np.ndarray, targets=(0.8, 0.9, 0.95)) -> dict:
    """Lowest exact score s whose kept set {score >= s} reaches each precision target. Candidates
    are the distinct scores, so a tie is kept or dropped as a whole; the cutoff is applied again
    to report the precision and share it actually gives."""
    score, correct = np.asarray(score, float), np.asarray(correct, bool)
    out = {}
    for target in targets:
        best = None
        for s in np.unique(score):  # ascending: the first that reaches the target keeps most
            keep = score >= s
            if correct[keep].mean() >= target:
                best = float(s)
                break
        if best is not None:
            keep = score >= best
            prec = float(correct[keep].mean())
            assert prec >= target, (target, best, prec)
            out[str(target)] = {"min_score": best, "precision": prec, "keeps": float(keep.mean()),
                                "n_kept": int(keep.sum())}
    return out


def calibration_table(score: pd.Series, correct: pd.Series) -> list:
    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cut = pd.cut(score, bins, right=False)
    return [{"from": float(b.left), "to": min(float(b.right), 1.0), "n": len(d),
             "precision": round(float(d.mean()), 4) if len(d) else None}
            for b, d in correct.groupby(cut, observed=False)]


def summarise(res: pd.DataFrame, idx2sign: dict) -> dict:
    acc = lambda d: {k: round(float(d[k].mean()), 4) for k in ("top1", "top3", "top5", "exact")} | {"n": len(d)}
    overall = {"combined": acc(res)} | {s: acc(d) for s, d in res.groupby("source")}

    table = calibration_table(res["p1"], res["top1"])
    thresholds = cutoffs(res["p1"].to_numpy(), res["top1"].to_numpy())
    em = pd.DataFrame([{"score": v, "correct": c == y} for y, cs, vs in
                       zip(res["sign"], res["emitted"], res["emitted_scores"]) for c, v in zip(cs, vs)])
    emitted = {"n_emitted": len(em), "clips_without_emission": int((res["emitted"].str.len() == 0).sum()),
               "by_score": calibration_table(em["score"], em["correct"]) if len(em) else [],
               "thresholds": cutoffs(em["score"].to_numpy(), em["correct"].to_numpy()) if len(em) else {}}

    signs = {}
    for y, d in res.groupby("sign"):
        wrong = d.loc[~d["top1"], "pred"].value_counts()
        signs[idx2sign[y]] = {
            "n": len(d), **{k: round(float(d[k].mean()), 4) for k in ("top1", "top3", "top5", "exact")},
            "confused_with": [[idx2sign[int(p)], round(n / len(d), 4)] for p, n in wrong.head(3).items()],
        }
    return {"overall": overall, "calibration": {"by_p1": table, "thresholds": thresholds}, "emitted": emitted,
            "signs": dict(sorted(signs.items()))}


def self_test() -> None:
    """Edge cases of the app-parity decode and of the cutoffs (`--self-test`, no data needed)."""
    f = lambda rows: torch.tensor(rows, dtype=torch.float32)
    # classes 0, 1 = signs, 2 = blank; 1e-8 apart = equal after float32 softmax
    assert [c for c, _ in emitted_scores(f([[0, 1e-8, -20]]), 2)] == [1], "near-tie sign: the logit argmax"
    assert emitted_scores(f([[0, -20, 1e-8]]), 2) == [], "near-tie with blank: no emission"
    # runs 0 (t 0-1), 1 (t 4): stretches 0..2 and 3..4 (halfway split); repeats after blank re-emit
    em = emitted_scores(f([[9, 0, 0], [9, 0, 0], [0, 0, 9], [0, 0, 9], [0, 9, 0], [0, 0, 9], [0, 9, 0]]), 2)
    assert [c for c, _ in em] == [0, 1, 1], em
    assert cutoffs(np.array([0.9, 0.9]), np.array([1, 0])) == {}, "a tie is never split"
    c = cutoffs(np.array([0.99, 0.9, 0.9]), np.array([1, 1, 0]), (0.8,))
    assert c["0.8"]["min_score"] == 0.99 and c["0.8"]["n_kept"] == 1, c
    c = cutoffs(np.array([0.1234567891, 0.5]), np.array([1, 1]), (0.8,))
    assert c["0.8"]["min_score"] == 0.1234567891, "cutoff kept unrounded"
    rng = np.random.default_rng(0)
    sc = np.round(rng.random(5000), 2)
    ok = rng.random(5000) < sc
    for t, v in cutoffs(sc, ok).items():
        assert ok[sc >= v["min_score"]].mean() >= float(t)
    print("sign_reliability self-test passed")


def main() -> None:
    import sys

    if "--self-test" in sys.argv:
        return self_test()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", default="data/asl-is-lmdb", type=Path)
    p.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb", type=Path)
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=7)
    p.add_argument("--wlasl-dir", type=Path, help="WLASL LMDB folder: also score its held-out signer half")
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--per-clip", type=Path, help="CSV: one row per clip")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    meta = json.loads(args.checkpoint.with_name("phrases_meta.json").read_text())
    sign2idx = load_base(args.data_dir / "sign_to_prediction_index_map.json")
    idx2sign = {v: k for k, v in sign2idx.items()}
    model = load_ctc_model(args.checkpoint, meta, len(sign2idx)).eval()

    releases = {"asl_signs": require_accepted(args.data_dir, args.lmdb_path)}
    if args.wlasl_dir:
        releases["wlasl"] = require_accepted(args.wlasl_dir, args.wlasl_dir / "wlasl.lmdb.mdb")
    df = pd.read_csv(args.data_dir / "train.csv")
    _, va = signer_fold_split(df, args.val_fold, args.n_folds)
    parts = [score_clips(model, df.iloc[va], args.lmdb_path, sign2idx, args.max_frames,
                         f"asl_signs_fold{args.val_fold}")]
    if args.wlasl_dir:
        w = pd.read_csv(args.wlasl_dir / "train.csv")
        w = w[w["participant_id"].isin(signer_split(w)["test"])]
        parts.append(score_clips(model, w, args.wlasl_dir / "wlasl.lmdb.mdb", sign2idx,
                                 args.max_frames, "wlasl_test_half"))
    res = pd.concat(parts, ignore_index=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "scoring": "pooled: per sign, max over the segment's frames of softmax(logits)[t, sign], "
                   "blank excluded; rank signs by it. p1 = the top sign's pooled probability. "
                   "Segment = hand-span trim, uniformly subsampled to <= 384 frames.",
        "dataset_releases": releases,
        "clips": "signers not seen in training: " + ", ".join(sorted(res["source"].unique())),
    } | summarise(res, idx2sign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    if args.per_clip:
        res.assign(sign=res["sign"].map(idx2sign), pred=res["pred"].map(idx2sign),
                   ranked=[" ".join(idx2sign[i] for i in t) for t in res["ranked"]],
                   emitted=[" ".join(idx2sign[i] for i in t) for t in res["emitted"]],
                   emitted_scores=[" ".join(repr(v) for v in t) for t in res["emitted_scores"]]
                   ).to_csv(args.per_clip, index=False)
    o = summary["overall"]["combined"]
    print(f"{o['n']} clips: top-1 {o['top1']:.3f} top-3 {o['top3']:.3f} top-5 {o['top5']:.3f} "
          f"exact {o['exact']:.3f} | pooled thresholds {summary['calibration']['thresholds']} | "
          f"emitted-sign thresholds {summary['emitted']['thresholds']} → {args.out}")


if __name__ == "__main__":
    main()
