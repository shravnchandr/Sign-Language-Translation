"""
Per-signer data diagnostics: what distinguishes signers the model gets wrong?

Signer-fold validation showed accuracy spreads of ~25 pts between signers in the
same fold (e.g. 0.49 vs 0.74). This script measures, per signer, properties of
the stored landmark data that could explain it — computed with the training
code's own LMDB keys and hand_presence(), on the raw stored coordinates (before
any augmentation), i.e. what the model actually sees:

  frames          clip length (frames)
  active          frames from the first to the last frame with a hand detected
  idle_lead/trail fraction of the clip before the first / after the last hand
                  detection (idle time around the sign itself)
  lh_det, rh_det  per-frame detection rate of each hand
  no_hand         fraction of frames with neither hand detected
  rh_dom          fraction of clips where the RH slot moves more — the model
                  mirrors these (HandDominanceModule); a signer far from the
                  population norm here signs with the other hand
  dom_ratio       dominant hand's share of wrist motion (0.5 two-handed .. 1 one-handed)
  speed           dominant-wrist xy motion per frame, in shoulder widths
  shoulder_w      shoulder width in image units (camera distance / framing proxy)
  pose_det, face_det  per-frame detection rate of pose / face landmarks

With --logs, each signer's final per-signer val accuracy is parsed from training
logs (the "per-signer val:" line after "Best val accuracy (deterministic)") and a
Spearman rank correlation of every statistic against accuracy is printed. With
few signers (e.g. 9 from 3 folds) treat correlations as leads, not conclusions.

Run from the repo root (CPU only; ~1–3 min at the default sample size):
  PYTHONPATH=research/models uv run python -m cnn_transformer.signer_diagnostics \\
      --logs 'logs/fold*.log'
"""

import argparse
import glob
import io
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .config import COORDS_PER_LM, FACE_START, LH_START, POSE_START, RH_START
from .data._cache_keys import lmdb_key
from .data.dataset import _open_lmdb_env
from .data.preprocessing import hand_presence

_C = COORDS_PER_LM


def _block_detected(block: torch.Tensor) -> torch.Tensor:
    """Per-frame detection for a landmark block, using the same rule as
    hand_presence(): stored data is ffill'd/zero-filled, so a missing frame is
    all-zero or a bit-exact repeat of the previous frame."""
    det = block.abs().sum(-1) > 0
    det[1:] &= ~(block[1:] == block[:-1]).all(-1)
    return det


def clip_stats(coords: torch.Tensor) -> dict:
    """Statistics for one stored clip (T, COORD_FEAT)."""
    filled, presence = hand_presence(coords)
    T = coords.shape[0]
    any_hand = (presence.sum(1) > 0).nonzero().flatten()
    if len(any_hand):
        first, last = int(any_hand[0]), int(any_hand[-1])
        active, lead, trail = last - first + 1, first / T, (T - 1 - last) / T
    else:  # no hand ever detected
        active, lead, trail = 0, float("nan"), float("nan")

    def wrist(start):
        # xy only: stored wrist z is -nose_z (a depth-frame artifact), which
        # otherwise dominates wrist motion energy (52–82% in sampled clips).
        return filled[:, start : start + 2]

    def energy(w):
        return float((w[1:] - w[:-1]).pow(2).sum(-1).mean()) if T > 1 else 0.0

    lh_e, rh_e = energy(wrist(LH_START)), energy(wrist(RH_START))
    ls = coords[:, POSE_START + 11 * _C : POSE_START + 11 * _C + 2]
    rs = coords[:, POSE_START + 12 * _C : POSE_START + 12 * _C + 2]
    shoulder_w = float((ls - rs).norm(dim=-1).mean())
    dom_e = max(lh_e, rh_e)
    return {
        "frames": T,
        "active": active,
        "idle_lead": lead,
        "idle_trail": trail,
        "lh_det": float(presence[:, 0].mean()),
        "rh_det": float(presence[:, 1].mean()),
        "no_hand": float((presence.sum(1) == 0).float().mean()),
        "rh_dom": float(rh_e > lh_e),
        "dom_ratio": dom_e / (lh_e + rh_e + 1e-12) if (lh_e + rh_e) > 0 else float("nan"),
        "speed": (dom_e ** 0.5) / shoulder_w if shoulder_w > 1e-6 else float("nan"),
        "shoulder_w": shoulder_w,
        "pose_det": float(_block_detected(coords[:, POSE_START:RH_START]).float().mean()),
        "face_det": float(_block_detected(coords[:, FACE_START:]).float().mean()),
    }


_BEST_RE = re.compile(r"Best val accuracy \(deterministic\)")
_PAIR_RE = re.compile(r"(\S+) (\d\.\d+)")


def parse_signer_accuracy(log_globs: list[str]) -> dict[str, float]:
    """Final per-signer val accuracy from training logs ({participant_id: acc}).
    Later files win if a signer appears in several."""
    acc: dict[str, float] = {}
    paths = sorted({p for g in log_globs for p in glob.glob(g)})
    for path in paths:
        lines = Path(path).read_text(errors="replace").replace("\r", "\n").splitlines()
        for i, line in enumerate(lines):
            if not _BEST_RE.search(line):
                continue
            # The per-signer line follows the deterministic summary; tolerate
            # other summary lines in between, stop at the next summary.
            for nxt in lines[i + 1 : i + 6]:
                if "Best val accuracy" in nxt:
                    break
                if "per-signer val:" in nxt:
                    body = nxt.split("per-signer val:", 1)[1].split("(spread")[0]
                    for pid, a in _PAIR_RE.findall(body):
                        acc[pid] = float(a)
                    break
    if log_globs and not acc:
        print(f"WARNING: no per-signer results found in {paths or log_globs} "
              "(runs must finish, and be made with per-signer logging)")
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/asl-is-lmdb", help="dir with train.csv")
    p.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb")
    p.add_argument("--per-signer", type=int, default=500,
                   help="clips sampled per signer (0 = all; ~4.5k per signer)")
    p.add_argument("--logs", nargs="*", default=[],
                   help="training log globs to pull per-signer val accuracy from")
    p.add_argument("--out", default="logs/signer_diagnostics.csv")
    args = p.parse_args()

    df = pd.read_csv(Path(args.data_dir) / "train.csv")
    df["participant_id"] = df["participant_id"].astype(str)
    if args.per_signer > 0:  # shuffle, then first k per signer: a seeded random sample
        df = df.sample(frac=1, random_state=0).groupby("participant_id").head(args.per_signer)

    rows, missing = [], 0
    with _open_lmdb_env(args.lmdb_path).begin(buffers=True) as txn:
        for pid, path in tqdm(zip(df["participant_id"], df["path"]), total=len(df), desc="clips"):
            val = txn.get(lmdb_key(path))
            if val is None:
                missing += 1
                continue
            coords = torch.load(io.BytesIO(bytes(val)), weights_only=True)
            rows.append({"participant_id": pid, **clip_stats(coords)})
    if missing:
        print(f"WARNING: {missing} clips not found in the LMDB (key/version mismatch?)")
    if not rows:
        raise SystemExit("no clips loaded")

    clips = pd.DataFrame(rows)
    stats = clips.groupby("participant_id").agg(
        n=("frames", "size"), **{c: (c, "mean") for c in clips.columns if c != "participant_id"}
    )
    stats.insert(1, "frames_med", clips.groupby("participant_id")["frames"].median())
    stats.insert(3, "active_med", clips.groupby("participant_id")["active"].median())

    acc = parse_signer_accuracy(args.logs)
    if acc:
        stats.insert(0, "val_acc", pd.Series(acc).reindex(stats.index))
        stats = stats.sort_values("val_acc", na_position="last")

    pd.set_option("display.width", 200)
    print(f"\nPer-signer statistics ({len(clips):,} clips, "
          f"{'all' if args.per_signer == 0 else f'≤{args.per_signer}'} per signer):\n")
    print(stats.round(3).to_string())

    pop = clips.drop(columns="participant_id").mean()
    print("\nPopulation mean: " + ", ".join(f"{k}={v:.3f}" for k, v in pop.items()))

    scored = stats.dropna(subset=["val_acc"]) if acc else pd.DataFrame()
    if len(scored) >= 3:
        cols = [c for c in stats.columns if c not in ("val_acc", "n")]
        rho = scored[cols].corrwith(scored["val_acc"], method="spearman").sort_values()
        print(f"\nSpearman rank correlation with val_acc over {len(scored)} signers "
              f"(|ρ| ≳ 0.7 worth a look at n≈9; leads, not conclusions):")
        for c, r in rho.items():
            print(f"  {c:<11} {r:+.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out)
    (out.with_suffix(".json")).write_text(json.dumps({"signer_acc": acc}, indent=2))
    print(f"\nSaved {out} (+ {out.with_suffix('.json').name})")


if __name__ == "__main__":
    main()
