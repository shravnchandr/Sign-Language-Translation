"""Score a checkpoint on another dataset's signers (WLASL LMDB from build_wlasl_lmdb.py).

Works for both model kinds:
  - phrase (CTC) checkpoints (phrases_meta.json next to the .pth): a clip is correct
    when the greedy decode of its hand-trimmed segment is exactly the one sign;
  - isolated classifiers (best_phase1.pth): top-1 / top-5 of the 250-way head.
The isolated path subsamples to --iso-max-frames before Δ1, as ASLDataset does at
evaluation; the phrase path trims to the hand span and caps at --max-frames.

Reports overall accuracy (aspect-corrected as stored, and with the correction undone
from `aspect_k` to show what framing costs), identical vs synonym glosses, clips with
vs without both hands, one-hand mode (less active hand removed) on two-hand clips with
an exact McNemar test, and per-signer spread. --out writes the numbers as JSON.

Usage:
  PYTHONPATH=research/models uv run python -m cnn_transformer.cross_dataset_eval \
    --checkpoint results/results_phrases01/checkpoints/phrases_fold0_graph/best_phrases.pth \
    --data-dir data/wlasl-lmdb --out results/phrases01_wlasl.json
"""

import argparse
import io
import json
import os
from math import comb
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch

from . import config as C
from .data._cache_keys import lmdb_key
from .error_review import load_model
from .export_onnx import clip_features, load_ctc_model

F = C.COORD_FEAT


def _with_velocity(pos: np.ndarray, pres: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel, pres], axis=1)


def _subsample(x: np.ndarray, max_frames: int) -> np.ndarray:
    if len(x) <= max_frames:
        return x
    keep = torch.linspace(0, len(x) - 1, max_frames).long().numpy()
    return _with_velocity(x[keep, :F], x[keep, -2:])


def _trim(x: np.ndarray) -> np.ndarray:
    nz = np.nonzero(x[:, -2:].sum(1) > 0)[0]
    return x[nz[0] : nz[-1] + 1] if len(nz) else x


def one_hand(x: np.ndarray) -> np.ndarray:
    """Drop the hand with less wrist motion (xy, frames where detected): never detected."""
    x = x.copy()
    energy = [(((x[1:, F + hs : F + hs + 2] ** 2).sum(1)) * (x[1:, -2 + h] > 0)).sum()
              for h, hs in enumerate((C.LH_START, C.RH_START))]
    drop = 0 if energy[0] < energy[1] else 1
    hs = (C.LH_START, C.RH_START)[drop]
    x[:, hs : hs + C.N_LH * C.COORDS_PER_LM] = 0.0
    x[:, -2 + drop] = 0.0
    return _with_velocity(x[:, :F], x[:, -2:])


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
    def predict(x: np.ndarray) -> tuple[bool, bool]:
        """(top-1 correct, top-5 correct) for the clip's label y (closure var)."""
        x = _subsample(_trim(x), args.max_frames) if phrase else _subsample(x, args.iso_max_frames)
        t = torch.from_numpy(x[None]).float().to(device)
        out = model(t, torch.ones(1, len(x), dtype=torch.bool, device=device))[0]
        if phrase:
            hyp, prev = [], None
            for c in out.argmax(-1).tolist():
                if c != prev and c != blank:
                    hyp.append(c)
                prev = c
            return hyp == [y], hyp == [y]
        return out.argmax().item() == y, y in out.topk(5).indices.tolist()

    df = pd.read_csv(args.data_dir / "train.csv")
    lmdb_path = args.lmdb_path or args.data_dir / "wlasl.lmdb.mdb"
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False,
                    map_size=os.path.getsize(lmdb_path) + (1 << 28))
    rows = []
    with env.begin() as txn:
        for r in df.itertuples():
            coords = torch.load(io.BytesIO(txn.get(lmdb_key(r.path))), weights_only=True)
            y = sign2idx[r.sign]
            x = clip_features(coords).numpy()
            raw = coords.clone()
            raw[:, 0::3] /= r.aspect_k
            raw[:, 2::3] /= r.aspect_k
            top1, top5 = predict(x)
            rows.append(dict(signer=r.participant_id, synonym=bool(r.synonym),
                             two_hands=r.both_hands > 0, correct=top1, top5=top5,
                             uncorrected=predict(clip_features(raw).numpy())[0],
                             one_hand=predict(one_hand(x))[0]))
    res = pd.DataFrame(rows)

    two = res[res["two_hands"]]
    per_signer = res.groupby("signer")["correct"].mean()
    summary = {
        "checkpoint": str(args.checkpoint),
        "kind": "phrase (single-sign exact decode)" if phrase else "isolated (top-1)",
        "clips": len(res), "signs": int(df["sign"].nunique()), "signers": int(res["signer"].nunique()),
        "accuracy": float(res["correct"].mean()),
        "accuracy_uncorrected": float(res["uncorrected"].mean()),
        "top5": None if phrase else float(res["top5"].mean()),
        "identical_gloss": float(res.loc[~res["synonym"], "correct"].mean()),
        "synonym_gloss": float(res.loc[res["synonym"], "correct"].mean()),
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


if __name__ == "__main__":
    main()
