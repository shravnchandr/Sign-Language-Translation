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
  signs        per sign: n, top1, top3, top5, exact, confused_with [[sign, share of clips]]

Usage:
  PYTHONPATH=research/models uv run python -m cnn_transformer.sign_reliability \
    --checkpoint <best_phrases.pth> --data-dir data/asl-is-lmdb \
    --lmdb-path data/asl-is-lmdb/is.lmdb.mdb --wlasl-dir data/wlasl-lmdb --out exports/phrases04/sign_reliability.json
"""

import argparse
import io
import json
import os
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from .cross_dataset_eval import _subsample, _trim
from .data._cache_keys import lmdb_key
from .data.build_wlasl_lmdb import signer_split
from .export_onnx import clip_features, load_ctc_model


def _env(path: Path):
    return lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                     map_size=os.path.getsize(path) + (1 << 28))


def score_clips(model, df: pd.DataFrame, lmdb_path: Path, sign2idx: dict, max_frames: int,
                source: str) -> pd.DataFrame:
    blank = len(sign2idx)
    rows = []
    with _env(lmdb_path).begin() as txn, torch.no_grad():
        for r in df.itertuples():
            coords = torch.load(io.BytesIO(txn.get(lmdb_key(r.path))), weights_only=True)
            x = _subsample(_trim(clip_features(coords).numpy()), max_frames)
            out = model(torch.from_numpy(x[None]).float(), torch.ones(1, len(x), dtype=torch.bool))[0]
            hyp, prev = [], None
            for c in out.argmax(-1).tolist():
                if c != prev and c != blank:
                    hyp.append(c)
                prev = c
            pooled = out.float().softmax(-1)[:, :blank].max(0).values
            top = pooled.topk(5)
            y = sign2idx[r.sign]
            rows.append(dict(source=source, signer=r.participant_id, sign=y,
                             ranked=top.indices.tolist(), p1=float(top.values[0]),
                             exact=hyp == [y]))
    res = pd.DataFrame(rows)
    for k in (1, 3, 5):
        res[f"top{k}"] = [y in t[:k] for y, t in zip(res["sign"], res["ranked"])]
    res["pred"] = [t[0] for t in res["ranked"]]
    return res


def summarise(res: pd.DataFrame, idx2sign: dict) -> dict:
    acc = lambda d: {k: round(float(d[k].mean()), 4) for k in ("top1", "top3", "top5", "exact")} | {"n": len(d)}
    overall = {"combined": acc(res)} | {s: acc(d) for s, d in res.groupby("source")}

    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cut = pd.cut(res["p1"], bins, right=False)
    table = [{"p1_from": float(b.left), "p1_to": min(float(b.right), 1.0), "n": len(d),
              "top1": round(float(d["top1"].mean()), 4) if len(d) else None}
             for b, d in res.groupby(cut, observed=False)]
    order = res.sort_values("p1", ascending=False)
    prec = order["top1"].cumsum().to_numpy() / np.arange(1, len(order) + 1)
    thresholds = {}
    for target in (0.8, 0.9, 0.95):
        ok = np.nonzero(prec >= target)[0]
        if len(ok):
            i = ok[-1]
            thresholds[str(target)] = {"min_p1": round(float(order["p1"].iloc[i]), 4),
                                       "keeps": round(float((i + 1) / len(order)), 4)}

    signs = {}
    for y, d in res.groupby("sign"):
        wrong = d.loc[~d["top1"], "pred"].value_counts()
        signs[idx2sign[y]] = {
            "n": len(d), **{k: round(float(d[k].mean()), 4) for k in ("top1", "top3", "top5", "exact")},
            "confused_with": [[idx2sign[int(p)], round(n / len(d), 4)] for p, n in wrong.head(3).items()],
        }
    return {"overall": overall, "calibration": {"by_p1": table, "thresholds": thresholds},
            "signs": dict(sorted(signs.items()))}


def main() -> None:
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
    sign2idx = json.loads((args.data_dir / "sign_to_prediction_index_map.json").read_text())
    idx2sign = {v: k for k, v in sign2idx.items()}
    model = load_ctc_model(args.checkpoint, meta, len(sign2idx)).eval()

    df = pd.read_csv(args.data_dir / "train.csv")
    _, va = list(GroupKFold(args.n_folds).split(df, groups=df["participant_id"]))[args.val_fold]
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
        "clips": "signers not seen in training: " + ", ".join(sorted(res["source"].unique())),
    } | summarise(res, idx2sign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    if args.per_clip:
        res.assign(sign=res["sign"].map(idx2sign), pred=res["pred"].map(idx2sign),
                   ranked=[" ".join(idx2sign[i] for i in t) for t in res["ranked"]]).to_csv(args.per_clip, index=False)
    o = summary["overall"]["combined"]
    print(f"{o['n']} clips: top-1 {o['top1']:.3f} top-3 {o['top3']:.3f} top-5 {o['top5']:.3f} "
          f"exact {o['exact']:.3f} | thresholds {summary['calibration']['thresholds']} → {args.out}")


if __name__ == "__main__":
    main()
