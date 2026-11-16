"""Two-hand exposure of a phrase-training mixture: what the model is actually fed, not what the
releases store. Builds the training phrases exactly as phrases.train does (its own parser and data
code: ASL Signs fold, extra sources, signer weights, trim, tempo, transitions, hold-fill, subsample to
--max-frames) for one or more --extra-frac mixtures, draws --n phrases of epoch 0 from each (same seed),
and counts model-input frames whose two presence flags are both 1 — overall and per source. Nothing is
trained; CPU only.

  PYTHONPATH=research/models python -m islr.phrases.exposure --n 4000 --mix 0.3,0.2 --mix 0.4,0.2 \
      --out <json> -- <phrases.train arguments, as in the frozen recipe>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..core import config


def build_train(args):
    """phrases.train's training phrase set for args (the lines of train() before the model)."""
    from ..core.data.dataset import get_data_loaders
    from ..core.data.vocab import check_map, load_base
    from . import train as T
    from .dataset import phrase_datasets

    sign2idx = load_base(Path(args.data_dir) / "sign_to_prediction_index_map.json")
    if args.vocab:
        if check_map(Path(args.vocab), extended=True):
            raise SystemExit(f"--vocab {args.vocab} does not extend the canonical map")
        sign2idx = json.load(open(args.vocab))
    idx2sign = {v: k for k, v in sign2idx.items()}
    tl, vl, _ = get_data_loaders(args.data_dir, lmdb_path=args.lmdb_path, num_workers=0,
                                 val_fold=args.val_fold, n_folds=args.n_folds, max_frames=1 << 20)
    source, weights = tl.dataset, None
    if args.extra_data_dir:
        source, weights = T.extra_source(args, tl.dataset, sign2idx)
    sources = list(getattr(source, "sources", [source]))
    for d, s in zip([args.data_dir, *args.extra_data_dir], sources):
        T.require_validated(d, s)  # the accepted releases only
    kw = dict(max_frames=args.max_frames, random_frac=args.random_frac, max_signs=args.max_signs,
              transition=(args.trans_min, args.trans_max), min_detected=args.min_detected,
              holdout_frac=args.holdout_frac)
    train_ds, _ = phrase_datasets(source, vl.dataset, idx2sign, args.train_phrases, 1, seed=args.seed,
                                  tempo=(args.tempo_min, args.tempo_max), signer_weights=weights,
                                  bone_aug=(args.bone_aug_prob, args.bone_aug_hand, args.bone_aug_bone), **kw)
    names = ["ASL Signs"] + [Path(d).name for d in args.extra_data_dir]
    owner = {str(p): n for n, s in zip(names, sources) for p in s.df["participant_id"].astype(str).unique()}
    shares = T.sampling_shares(train_ds, sources, [1 - sum(args.extra_frac)] + list(args.extra_frac)
                               if args.extra_data_dir else [1.0])
    return train_ds, owner, names, shares


def measure(train_ds, owner, names, n: int) -> dict:
    ps = config.PRESENCE_START
    tot = {k: {"phrases": 0, "frames": 0, "both_hand_frames": 0} for k in names}
    for i in range(min(n, len(train_ds))):
        x, _, _ = train_ds[i]
        both = int(((x[:, ps] > 0) & (x[:, ps + 1] > 0)).sum())
        t = tot[owner[train_ds.specs[i][0]]]
        t["phrases"] += 1
        t["frames"] += int(x.shape[0])
        t["both_hand_frames"] += both
    frames = sum(t["frames"] for t in tot.values())
    both = sum(t["both_hand_frames"] for t in tot.values())
    for t in tot.values():
        t["both_share"] = t["both_hand_frames"] / max(t["frames"], 1)
        t["frame_share"] = t["frames"] / max(frames, 1)
        t["share_of_both_hand_frames"] = t["both_hand_frames"] / max(both, 1)
    return {"phrases": sum(t["phrases"] for t in tot.values()), "frames": frames,
            "both_hand_frames": both, "both_share": both / max(frames, 1), "by_source": tot}


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: exposure [--n N] --mix f1,f2 [--mix ...] [--out J] -- <phrases.train args>")
    cut = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=4000, help="phrases drawn per mixture (epoch 0, same seed)")
    p.add_argument("--mix", action="append", required=True, help="comma-separated --extra-frac values")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv[:cut])

    from . import train as T

    got = {}
    old_train, old_argv = T.train, sys.argv
    T.train, sys.argv = (lambda args: got.update(args=args)), ["phrases.train", *argv[cut + 1:], "--out-dir", "unused"]
    try:
        T.main()
    finally:
        T.train, sys.argv = old_train, old_argv
    args = got["args"]
    res = {}
    for m in a.mix:
        args.extra_frac = [float(v) for v in m.split(",") if v]
        train_ds, owner, names, shares = build_train(args)
        r = measure(train_ds, owner, names, a.n)
        r["requested_phrase_shares"] = dict(zip(names, shares))
        res[m] = r
        print(f"mix {m}: {r['phrases']} phrases, {r['frames']:,} input frames, both hands {r['both_share']:.2%}")
        for k, t in r["by_source"].items():
            print(f"  {k:24s} phrases {t['phrases']:5d}  frames {t['frame_share']:6.1%}  both-hand share "
                  f"{t['both_share']:6.2%}  of all both-hand frames {t['share_of_both_hand_frames']:6.1%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
