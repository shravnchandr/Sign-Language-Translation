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
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import config as C
from ..core.data._cache_keys import lmdb_key
from ..core.data.build_wlasl_lmdb import signer_split
from ..core.data.vocab import load_map
from ..core.data.dataset import _open_lmdb_env
from ..core.data.validate import require_accepted
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


def partition_rows(df: pd.DataFrame, release: dict, spec: str, score_final: bool) -> tuple[pd.DataFrame, dict]:
    """The rows of one part of a committed signer partition (popsign_test_split.json format), refusing a
    manifest written for another release or one that does not account for every signer in it."""
    path, _, part = spec.rpartition(":")
    if not path or part not in ("development", "final"):
        raise SystemExit(f"--partition {spec!r}: expected MANIFEST:development or MANIFEST:final")
    if part == "final" and not score_final:
        raise SystemExit("--partition …:final is the sealed part; pass --score-final to score it (once)")
    raw = Path(path).read_bytes()
    m = json.loads(raw)
    want = {"content": m["release"]["content_digest"], "csv": m["release"]["csv_digest"]}
    got = {k: release.get(k) for k in want}
    if got != want:
        raise SystemExit(f"--partition {path} was written for release {want}, the opened release is {got}")
    parts = {"development": m["development"], "final": m["final"], "excluded": list(m["excluded"])}
    ids = {k: {int(x) for x in v} for k, v in parts.items()}
    every = [x for v in ids.values() for x in v]
    if len(every) != len(set(every)):
        raise SystemExit(f"--partition {path}: a signer is in more than one part")
    present = set(df["participant_id"].astype(int))
    if present != set(every):
        raise SystemExit(f"--partition {path} does not match the release's signers: missing from the manifest "
                         f"{sorted(present - set(every))}, not in the release {sorted(set(every) - present)}")
    sel = df[df["participant_id"].astype(int).isin(ids[part])]
    signers = sorted(ids[part])
    print(f"partition {Path(path).name}:{part} — {len(sel)} clips, signers {signers} "
          f"(excluded from every part: {sorted(ids['excluded'])})")
    return sel, {"manifest": str(path), "sha256": hashlib.sha256(raw).hexdigest()[:16], "part": part,
                 "signers": signers, "excluded_signers": sorted(ids["excluded"]), "clips": len(sel)}


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
    p.add_argument("--val-fold", type=int, help="ASL Signs: score the held-out signers of this frozen fold "
                   "(dataset.signer_fold_split; no aspect correction, both_hands from stored presence)")
    p.add_argument("--partition", help="MANIFEST:PART — the signers of one part of a committed signer partition "
                   "(e.g. core/data/popsign_test_split.json:development); the manifest's release digests and "
                   "signer list must match the opened release")
    p.add_argument("--score-final", action="store_true", help="allow --partition …:final (the sealed part: score "
                   "once, for a model already chosen on development)")
    args = p.parse_args()
    if args.val_fold is not None and (args.signers != "all" or args.split):
        raise SystemExit("--val-fold selects ASL Signs signers; it does not combine with --signers / --split")
    if args.partition and (args.signers != "all" or args.split or args.val_fold is not None):
        raise SystemExit("--partition selects the signers itself; it does not combine with --signers / --split / --val-fold")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign2idx = load_map(args.sign_map or args.data_dir / "sign_to_prediction_index_map.json", extended=True)
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
    fold_signers = None
    if args.val_fold is not None:  # ASL Signs: the frozen fold's held-out signers, as the phrase validation
        from ..core.data.dataset import _FOLDS, signer_fold_split

        folds = json.loads(_FOLDS.read_text())
        if not 0 <= args.val_fold < folds["n_folds"]:
            raise SystemExit(f"--val-fold {args.val_fold}: the frozen ASL Signs folds are 0..{folds['n_folds'] - 1}")
        unknown = sorted(set(df["participant_id"].astype(str)) - set(folds["folds"]))
        if unknown or "source" in df:
            raise SystemExit(f"--val-fold needs the ASL Signs release (signers not in the frozen folds: {unknown[:5]})")
        df = df.iloc[signer_fold_split(df, args.val_fold)[1]]
        fold_signers = sorted(df["participant_id"].astype(str).unique())
        df = df.assign(synonym=False, aspect_k=1.0, ratio_raw=np.nan, fps=30, source="aslsigns", both_hands=-1)
    lmdb_path = args.lmdb_path or args.data_dir / "wlasl.lmdb.mdb"
    release = require_accepted(args.data_dir, lmdb_path)  # scores only on a validated dataset release
    partition = None
    if args.partition:
        df, partition = partition_rows(df, release, args.partition, args.score_final)
    env = _open_lmdb_env(str(lmdb_path))  # the handle require_accepted opened (one env per process)
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
                             split=getattr(r, "split", None),
                             two_hands=bool(pres.all(1).any()) if r.both_hands < 0 else r.both_hands > 0,
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
        "val_fold": args.val_fold, "fold_signers": fold_signers,
        "partition": partition,
        "kind": "phrase (single-sign exact decode)" if phrase else "isolated (top-1)",
        "dataset_release": release,
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
        # never a detected hand: kept and scored like any clip (the model sees body / face only)
        "no_hand_clips": {"n": int((res["hand_frac"] == 0).sum()),
                          "accuracy": float(res.loc[res["hand_frac"] == 0, "correct"].mean()) if (res["hand_frac"] == 0).any() else None},
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
