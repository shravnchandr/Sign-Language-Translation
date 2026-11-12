"""Full validation breakdown of a phrase (CTC) checkpoint (CPU).

Rebuilds the fixed val phrases exactly as train_phrases.py does (same seed, fold,
phrase settings and length-sorted order — pass the run's values if not defaults),
then reports:
  - WER / exact / substitution, deletion, insertion rates by phrase length, kind,
    signer and length × signer
  - deletions on 4–5-sign phrases with vs without the --max-frames downsample
  - emission timing: whether the frame where CTC emits each matched sign lies
    inside that sign's own clip span (a proxy, not a boundary metric)
  - optionally (--iso-checkpoint) a controlled single-sign comparison: the isolated
    classifier on the identical trimmed clips (its own 128-frame cap), with McNemar

Usage:
  PYTHONPATH=research/models uv run python -m islr.phrases.breakdown \
    --checkpoint results/results_phrases01/checkpoints/phrases_fold0_graph/best_phrases.pth \
    --iso-checkpoint results/results_fold0_graph/checkpoints/fold0_graph/best_phase1.pth \
    --out results/results_phrases01/breakdown.txt
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..core.data.dataset import get_data_loaders
from .dataset import phrase_datasets
from ..isolated.error_review import load_model
from ..export.export_onnx import load_ctc_model


def align(ref: list, hyp: list) -> list:
    """Levenshtein alignment → [(op, ref_i, hyp_j)], op in C/S/D/I."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    return ops[::-1]


def greedy_with_frames(best: list, blank: int) -> tuple[list, list]:
    hyp, frames, prev = [], [], None
    for f, c in enumerate(best):
        if c != prev and c != blank:
            hyp.append(c)
            frames.append(f)
        prev = c
    return hyp, frames


def row(c: Counter) -> str:
    n = max(c["N"], 1)
    return (f"phrases {c['phrases']:5d} signs {c['N']:5d} WER {(c['S'] + c['D'] + c['I']) / n:.3f} "
            f"exact {c['exact'] / max(c['phrases'], 1):.3f} | S {c['S'] / n:.3f} "
            f"D {c['D'] / n:.3f} I {c['I'] / n:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--meta", type=Path, help="default: phrases_meta.json next to the checkpoint")
    p.add_argument("--iso-checkpoint", help="isolated classifier for the single-sign comparison")
    p.add_argument("--iso-n-heads", type=int, default=4)
    p.add_argument("--iso-max-frames", type=int, default=128)
    p.add_argument("--data-dir", default="data/asl-is-lmdb")
    p.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb")
    p.add_argument("--n-folds", type=int, default=7)
    # Must match the training run (train_phrases.py defaults).
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-phrases", type=int, default=3000)
    p.add_argument("--random-frac", type=float, default=0.3)
    p.add_argument("--max-signs", type=int, default=5)
    p.add_argument("--trans-min", type=int, default=3)
    p.add_argument("--trans-max", type=int, default=8)
    p.add_argument("--min-detected", type=int, default=4)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    meta = json.loads((args.meta or args.checkpoint.with_name("phrases_meta.json")).read_text())
    sign2idx = json.load(open(Path(args.data_dir) / "sign_to_prediction_index_map.json"))
    idx2sign = {v: k for k, v in sign2idx.items()}
    blank = len(sign2idx)

    tl, vl, _ = get_data_loaders(args.data_dir, lmdb_path=args.lmdb_path, num_workers=0,
                                 val_fold=meta["val_fold"], n_folds=args.n_folds,
                                 max_frames=1 << 20)
    _, val = phrase_datasets(
        tl.dataset, vl.dataset, idx2sign, 1, args.val_phrases, seed=args.seed,
        max_frames=args.max_frames, random_frac=args.random_frac, max_signs=args.max_signs,
        transition=(args.trans_min, args.trans_max), min_detected=args.min_detected,
        holdout_frac=args.holdout_frac,
    )
    order = np.argsort(val.lengths(), kind="stable")  # as train_phrases: idx seeds the build
    val.specs = [val.specs[i] for i in order]

    ctc = load_ctc_model(args.checkpoint, meta, blank)
    iso = load_model(args.iso_checkpoint, args.iso_n_heads)[0] if args.iso_checkpoint else None

    by = {k: defaultdict(Counter) for k in ("length", "kind", "signer", "length x signer")}
    long_del = defaultdict(Counter)  # (downsampled, signer) → counts, 4–5-sign phrases
    in_span = Counter()
    single = Counter()
    for idx in range(len(val)):
        x, _, ids = val[idx]
        signer, _, _, clips, trans, kind = val.specs[idx]
        # Clip spans: replay build()'s rng (val clips are only swapped, never resampled).
        rng = np.random.default_rng([val.seed, val.epoch, idx, 1])
        lens = [val._clip(rng, signer, s, j)[0].shape[0] for s, j in zip(ids, clips)]
        spans, t = [], 0
        for k, L in enumerate(lens):
            spans.append((t, t + L))
            t += L + (trans[k] if k < len(trans) else 0)
        downsampled = t > args.max_frames
        keep = (torch.linspace(0, t - 1, args.max_frames).long() if downsampled
                else torch.arange(t))

        with torch.no_grad():
            best = ctc(x[None], torch.ones(1, x.shape[0], dtype=torch.bool))[0].argmax(-1)
        hyp, frames = greedy_with_frames(best.tolist(), blank)
        frames = [int(keep[f]) for f in frames]
        ops = align(list(ids), hyp)

        c = Counter(o[0] for o in ops)
        c.update(N=len(ids), exact=int(hyp == list(ids)), phrases=1)
        for key, k in (("length", len(ids)), ("kind", kind), ("signer", signer),
                       ("length x signer", (len(ids), signer))):
            by[key][k].update(c)
        if len(ids) >= 4:
            long_del[(downsampled, signer)].update(c)
        for op, ri, hj in ops:
            if op in "CS":
                a, b = spans[ri]
                in_span[(op, a <= frames[hj] < b)] += 1

        if iso is not None and len(ids) == 1:
            xi = x
            if x.shape[0] > args.iso_max_frames:  # the classifier's own cap; Δ1 follows it
                xi = x[torch.linspace(0, x.shape[0] - 1, args.iso_max_frames).long()].clone()
                n = (xi.shape[-1] - 2) // 2
                xi[0, n : 2 * n] = 0
                xi[1:, n : 2 * n] = xi[1:, :n] - xi[:-1, :n]
            with torch.no_grad():
                pred = iso(xi[None], torch.ones(1, xi.shape[0], dtype=torch.bool)).argmax(-1).item()
            iso_ok, ctc_ok = pred == ids[0], hyp == list(ids)
            single.update(n=1, iso=iso_ok, ctc=ctc_ok, both=iso_ok and ctc_ok,
                          iso_only=iso_ok and not ctc_ok, ctc_only=ctc_ok and not iso_ok)
        if idx % 500 == 0:
            print(f"{idx}/{len(val)}", flush=True)

    lines = []
    for key, groups in by.items():
        lines.append(f"-- by {key}")
        lines += [f"{k}: {row(c)}" for k, c in sorted(groups.items())]
    lines.append(f"-- total: {row(sum(by['length'].values(), Counter()))}")
    lines.append(f"-- 4–5-sign phrases, downsampled to {args.max_frames} vs full length")
    lines += [f"{'downsampled' if d else 'full'} {s}: {row(c)}"
              for (d, s), c in sorted(long_del.items())]
    lines.append(
        f"-- emission frame inside own clip span: correct {in_span[('C', True)]}/"
        f"{in_span[('C', True)] + in_span[('C', False)]}, substituted {in_span[('S', True)]}/"
        f"{in_span[('S', True)] + in_span[('S', False)]}")
    if single["n"]:
        b, c = single["iso_only"], single["ctc_only"]
        chi2 = (b - c) ** 2 / (b + c) if b + c else 0.0
        lines.append(
            f"-- single sign, same trimmed clips (n {single['n']}): isolated {single['iso']} "
            f"CTC {single['ctc']} | both {single['both']} isolated-only {b} CTC-only {c} | "
            f"McNemar chi2 {chi2:.2f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
