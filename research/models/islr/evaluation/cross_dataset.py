"""Score a checkpoint on another dataset's signers (WLASL or ASL Citizen LMDB).

Inputs: the LMDB + train.csv of `build_wlasl_lmdb.py` or `build_aslcitizen_lmdb.py` (research
use only; pass --lmdb-path, and --split to pick an official ASL Citizen split).

Works for both model kinds:
  - phrase (CTC) checkpoints (phrases_meta.json next to the .pth): a clip is correct
    when the greedy decode of its hand-trimmed segment is exactly the one sign. Also
    scored like a classifier ("pooled": each sign's best per-frame log-probability
    over the clip, blank excluded → top-1 / top-5) and leniently (decode contains it);
  - isolated classifiers (best_phase1.pth): top-1 / top-5 of the 250-way head.
The isolated path subsamples to --iso-max-frames before Δ1, as ASLDataset does at
evaluation; the phrase path trims to the hand span and caps at --max-frames.

Reports overall accuracy (aspect-corrected as stored, and with the correction undone
from `aspect_k` to show what framing costs), identical vs synonym glosses, clips with
vs without both hands, one-hand mode (less active hand removed) on two-hand clips with
an exact McNemar test, per-signer spread and, when train.csv has a `kind` column (ASL
Citizen: identical / variant / synonym), accuracy per gloss kind. --out writes the numbers as JSON,
--per-clip one row per clip (CSV) for paired comparisons between checkpoints.

Usage:
  PYTHONPATH=research/models uv run python -m islr.evaluation.cross_dataset \
    --checkpoint results/phrases/phrases04/best_phrases.pth \
    --data-dir data/wlasl-lmdb --out results/phrases01_wlasl.json
"""

import argparse
import json
import os
from math import comb
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch

from ..core import config as C
from ..core.data._cache_keys import lmdb_key
from ..core.data.build_wlasl_lmdb import signer_split
from ..isolated.error_review import load_model
from ..core.data.contract import clip_input, load_clip
from ..export.export_onnx import load_ctc_model

F = C.COORD_FEAT


def one_hand(coords: torch.Tensor, presence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the hand with less wrist motion (xy, frames where detected): never detected."""
    coords, presence = coords.clone(), presence.clone()
    vel = coords[1:] - coords[:-1]
    energy = [float(((vel[:, hs : hs + 2] ** 2).sum(1) * (presence[1:, h] > 0)).sum())
              for h, hs in enumerate((C.LH_START, C.RH_START))]
    drop = 0 if energy[0] < energy[1] else 1
    hs = (C.LH_START, C.RH_START)[drop]
    coords[:, hs : hs + C.N_LH * C.COORDS_PER_LM] = 0.0
    presence[:, drop] = 0.0
    return coords, presence


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    up, down = int((~a & b).sum()), int((a & ~b).sum())
    n = up + down
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2**n) if n else 1.0
    return {"before": float(a.mean()), "after": float(b.mean()), "gained": up, "lost": down, "p": p}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="folder with wlasl.lmdb.mdb (or --lmdb-path) and train.csv")
    p.add_argument("--lmdb-path", type=Path)
    p.add_argument("--sign-map", type=Path, help="default: <data-dir>/sign_to_prediction_index_map.json")
    p.add_argument("--n-heads", type=int, default=4, help="isolated checkpoints only")
    p.add_argument("--iso-max-frames", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--out", type=Path)
    p.add_argument("--per-clip", type=Path, help="CSV: one row per clip")
    p.add_argument("--signers", choices=["all", "train", "test"], default="all",
                   help="WLASL signer half (build_wlasl_lmdb.signer_split); use test for a "
                        "model trained with --extra-data-dir")
    p.add_argument("--split", help="keep rows whose `split` column equals this (ASL Citizen: train / val / test)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign2idx = json.loads((args.sign_map or args.data_dir / "sign_to_prediction_index_map.json").read_text())
    meta_path = args.checkpoint.with_name("phrases_meta.json")
    phrase = meta_path.exists()
    if phrase:
        model = load_ctc_model(args.checkpoint, json.loads(meta_path.read_text()), len(sign2idx))
    else:
        model = load_model(str(args.checkpoint), args.n_heads)[0]
    model = model.to(device).eval()
    blank = len(sign2idx)

    @torch.no_grad()
    def predict(coords: torch.Tensor, presence: torch.Tensor) -> dict:
        """Scores for the clip's label y (closure var): correct, top5 [, pooled, decode]. Features by
        the data contract: phrase models segment to the hand span, isolated models do not."""
        x = clip_input(coords, presence, args.max_frames if phrase else args.iso_max_frames, trim=phrase)
        t = x[None].float().to(device)
        out = model(t, torch.ones(1, len(x), dtype=torch.bool, device=device))[0]
        if not phrase:
            return {"correct": out.argmax().item() == y, "top5": y in out.topk(5).indices.tolist(),
                    "pred": out.argmax().item()}
        hyp, prev = [], None
        for c in out.argmax(-1).tolist():
            if c != prev and c != blank:
                hyp.append(c)
            prev = c
        pooled = out.float().log_softmax(-1)[:, :blank].max(0).values
        top5 = pooled.topk(5).indices.tolist()
        return {"correct": hyp == [y], "top5": y in top5, "pooled": top5[0] == y,
                "contains": y in hyp, "n_tokens": len(hyp), "pred": top5[0]}

    df = pd.read_csv(args.data_dir / "train.csv")
    if args.signers != "all":
        df = df[df["participant_id"].isin(signer_split(df)[args.signers])]
    if args.split:
        df = df[df["split"] == args.split]
    lmdb_path = args.lmdb_path or args.data_dir / "wlasl.lmdb.mdb"
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False,
                    map_size=os.path.getsize(lmdb_path) + (1 << 28))
    rows = []
    with env.begin() as txn:
        for r in df.itertuples():
            coords, presence = load_clip(txn.get(lmdb_key(r.path)))
            y = sign2idx[r.sign]
            raw = coords.clone()
            raw[:, 0::3] /= r.aspect_k
            raw[:, 2::3] /= r.aspect_k
            pres = presence.numpy() > 0
            rows.append(dict(path=r.path, sign=r.sign, signer=r.participant_id,
                             synonym=bool(r.synonym),
                             kind=getattr(r, "kind", "synonym" if r.synonym else "identical"),
                             split=getattr(r, "split", None), two_hands=r.both_hands > 0,
                             n_frames=len(coords), hand_frac=float(pres.any(1).mean()),
                             aspect_k=r.aspect_k, ratio_raw=r.ratio_raw, fps=r.fps,
                             source=r.source, **predict(coords, presence),
                             uncorrected=predict(raw, presence)["correct"],
                             one_hand=predict(*one_hand(coords, presence))["correct"]))
    res = pd.DataFrame(rows)

    two = res[res["two_hands"]]
    per_signer = res.groupby("signer")["correct"].mean()
    summary = {
        "checkpoint": str(args.checkpoint),
        "wlasl_signers": args.signers,
        "split": args.split,
        "kind": "phrase (single-sign exact decode)" if phrase else "isolated (top-1)",
        "clips": len(res), "signs": int(df["sign"].nunique()), "signers": int(res["signer"].nunique()),
        "accuracy": float(res["correct"].mean()),
        "accuracy_uncorrected": float(res["uncorrected"].mean()),
        "top5": float(res["top5"].mean()),
        "pooled_top1": float(res["pooled"].mean()) if phrase else None,
        "decode_contains": float(res["contains"].mean()) if phrase else None,
        "decode_tokens": {int(k): int(v) for k, v in res["n_tokens"].value_counts().sort_index().items()} if phrase else None,
        "identical_gloss": float(res.loc[~res["synonym"], "correct"].mean()),
        "synonym_gloss": float(res.loc[res["synonym"], "correct"].mean()),
        "by_kind": {k: {"n": len(g), "accuracy": float(g["correct"].mean()),
                        "top5": float(g["top5"].mean()),
                        "pooled_top1": float(g["pooled"].mean()) if phrase else None}
                    for k, g in res.groupby("kind")},
        "two_hand_clips": {"n": len(two), "accuracy": float(two["correct"].mean())},
        "one_hand_clips": {"n": int((~res["two_hands"]).sum()),
                           "accuracy": float(res.loc[~res["two_hands"], "correct"].mean())},
        "one_hand_mode_on_two_hand_clips": mcnemar(two["correct"], two["one_hand"]),
        "per_signer": {"median": float(per_signer.median()), "p10": float(per_signer.quantile(0.1)),
                       "p90": float(per_signer.quantile(0.9))},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    if args.per_clip:
        args.per_clip.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.per_clip, index=False)


if __name__ == "__main__":
    main()
