"""
Build a self-contained HTML kit for reviewing a model's validation errors.

Question it answers: for a given mistake, is the correct sign actually visible
in the landmark data? Each item shows the misclassified clip (the "query") and
two unlabelled reference sets — real training clips of the true sign and of the
sign the model predicted, from other signers, in random order. The reviewer
picks which set the query matches; only then are the labels revealed.

  - picks the true sign reliably  → the information is in the landmarks; the
    model is failing to use it (a learner problem)
  - picks the predicted sign / cannot tell → the landmarks do not separate the
    two signs (ambiguous signs, labels, or lost tracking)
  - "hand missing / frozen at the key moment" → a tracking problem

No ASL knowledge is needed for the comparison. An optional first step lets a
fluent signer name the query sign before seeing the references (stronger).

Errors are sampled evenly across the validation signers, half from the most
frequent confusion pairs. Hand frames that MediaPipe did not detect (held
positions in the stored data) are drawn faded, so tracking gaps are visible.

Run from the repo root (CPU; a few minutes for 14k clips):
  PYTHONPATH=research/models uv run python -m islr.isolated.error_review \\
      --checkpoint results/isolated/fold0_graph/best_phase1.pth \\
      --also results/isolated/fold0_fixed/best_phase1.pth \\
      --out results/evaluation/error_review/fold0_graph_errors.html
"""

import argparse
import base64
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..core.config import COORDS_PER_LM
from ..core.data.dataset import collate_batch, get_data_loaders
from ..core.data.preprocessing import hand_presence
from ..core.model.graph_conformer import GraphConformer
from ..core.model.landmark_conformer import LandmarkConformer

# Stored layout: [LH 0–20 | pose 21–53 | RH 54–74 | face 75–130] landmarks.
# Display order per frame: LH 21, RH 21, pose 0–24 (upper body), face 56 = 123 points.
_DISPLAY_LMS = list(range(0, 21)) + list(range(54, 75)) + list(range(21, 46)) + list(range(75, 131))
_MAX_FRAMES = 96


def load_model(path: str, n_heads: int = 4):
    """Rebuild the model from a best_phase1.pth state dict (architecture inferred from keys).

    Head count is not recoverable from the weights (the attention projections have the
    same shapes for any divisor of d_model) and fine-tune checkpoints do not record it,
    so it must match the run's --n-heads (training default 4).
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    d_model = sd["head.3.weight"].shape[1]
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    kw = dict(
        num_classes=sd["head.3.weight"].shape[0],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_signers=sd["signer_disc.net.3.weight"].shape[0] if "signer_disc.net.3.weight" in sd else 0,
        supcon_dim=sd["proj_head.2.weight"].shape[0] if "proj_head.2.weight" in sd else 0,
    )
    if "hand_graph.joint_emb" in sd:
        kw["graph_hidden"] = sd["hand_graph.joint_emb"].shape[1]
        kw["graph_layers"] = len({k.split(".")[2] for k in sd if k.startswith("hand_graph.layers.")})
        model = GraphConformer(**kw)
    else:
        model = LandmarkConformer(**kw)
    model.load_state_dict(sd, strict=True)
    model.train(False)
    return model, ("graph" if "hand_graph.joint_emb" in sd else "conformer")


@torch.no_grad()
def predict(model, dataset, batch_size=64):
    """Softmax probabilities for every clip in dataset order (length-sorted batches)."""
    order = np.argsort(dataset.lengths)
    probs = np.zeros((len(dataset), model.head[-1].out_features), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        x, mask, _, _ = collate_batch([dataset[j] for j in idx])
        probs[idx] = torch.softmax(model(x, mask).float(), -1).numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  {min(i + batch_size, len(order)):,}/{len(order):,} clips", flush=True)
    return probs


def clip_payload(dataset, idx):
    """Stored landmarks of one clip → compact display record (xy, uint16, presence bits)."""
    coords = dataset._load_coords(idx)
    filled, presence = hand_presence(coords)  # presence: (T, 2) lh, rh
    presence = presence.numpy() > 0
    ever = presence.max(0)  # hand detected at least once — over all frames, before display sampling
    T = filled.shape[0]
    if T > _MAX_FRAMES:
        # Show one real frame per window, with its own flags (a held pose never looks
        # detected). Within a window prefer frames with more hands detected, then the
        # hand detected in fewer of the window's frames (brief detections survive),
        # then the frame nearest the window centre.
        edges = np.linspace(0, T, _MAX_FRAMES + 1).round().astype(int)
        keep = []
        for a, b in zip(edges[:-1], edges[1:]):
            win = presence[a:b].astype(float)
            rare = (win / np.maximum(win.sum(0), 1)).sum(1)
            centre = -np.abs(np.arange(b - a) - (b - a - 1) / 2)
            keep.append(a + int(np.lexsort((centre, rare, win.sum(1)))[-1]))
        filled, presence = filled[keep], presence[keep]
    pts = filled.numpy().reshape(filled.shape[0], -1, COORDS_PER_LM)[:, _DISPLAY_LMS, :2]  # (T, 123, 2)
    real = np.ones(pts.shape[1], dtype=bool)
    real[:21], real[21:42] = ever[0], ever[1]
    sel = pts[:, real].reshape(-1, 2)
    lo, hi = sel.min(0), sel.max(0)
    span = np.maximum(hi - lo, 1e-6)
    q = np.clip(np.round((pts - lo) / span * 65535), 0, 65535).astype("<u2")
    bits = presence[:, 0].astype(np.uint8) | (presence[:, 1].astype(np.uint8) << 1)
    return {
        "t": int(pts.shape[0]),
        "lo": [float(lo[0]), float(lo[1])],
        "span": [float(span[0]), float(span[1])],
        "xy": base64.b64encode(q.tobytes()).decode(),
        "p": base64.b64encode(bits.tobytes()).decode(),
        "ever": [bool(ever[0]), bool(ever[1])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model whose errors are reviewed")
    ap.add_argument("--also", nargs="*", default=[], help="other checkpoints, shown for context after reveal")
    ap.add_argument("--data-dir", default="data/asl-is-lmdb")
    ap.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=100, help="errors to review")
    ap.add_argument("--refs", type=int, default=2, help="reference clips per side")
    ap.add_argument("--top-pairs", type=int, default=20, help="confusion pairs counted as 'frequent'")
    ap.add_argument("--min-ref-hand", type=float, default=0.5,
                    help="reference clips need a detected hand in at least this fraction of frames")
    ap.add_argument("--min-ref-frames", type=int, default=8, help="… and at least this many detected frames")
    ap.add_argument("--n-heads", type=int, default=4, help="attention heads of every checkpoint (= training --n-heads)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/evaluation/error_review/error_review.html")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sign2idx = json.load(open(Path(args.data_dir) / "sign_to_prediction_index_map.json"))
    idx2sign = {v: k for k, v in sign2idx.items()}
    train_loader, val_loader, _ = get_data_loaders(
        args.data_dir, lmdb_path=args.lmdb_path, num_workers=0, val_fold=args.val_fold
    )
    train_ds, val_ds = train_loader.dataset, val_loader.dataset
    y = val_ds.df["sign"].to_numpy()
    signer = val_ds.df["participant_id"].astype(str).to_numpy()

    model, arch = load_model(args.checkpoint, args.n_heads)
    print(f"Predicting {len(val_ds):,} val clips with {args.checkpoint} ({arch}, {args.n_heads} heads)")
    probs = predict(model, val_ds)
    pred = probs.argmax(1)
    print(f"  accuracy {np.mean(pred == y):.4f}")
    others = {}
    for path in args.also:
        m, a = load_model(path, args.n_heads)
        name = Path(path).parent.name
        print(f"Predicting with {path} ({a})")
        others[name] = predict(m, val_ds).argmax(1)
        print(f"  accuracy {np.mean(others[name] == y):.4f}")

    # --- choose errors: even across signers, half from frequent confusion pairs ---
    err = np.flatnonzero(pred != y)
    pair = lambda i: tuple(sorted((int(y[i]), int(pred[i]))))
    freq = {p for p, _ in Counter(pair(i) for i in err).most_common(args.top_pairs)}
    signers = sorted(set(signer))
    chosen = []
    for k, s in enumerate(signers):
        quota = args.n // len(signers) + (1 if k < args.n % len(signers) else 0)
        pool = [int(i) for i in err if signer[i] == s]
        rng.shuffle(pool)
        in_pair = [i for i in pool if pair(i) in freq][: quota // 2]
        rest = [i for i in pool if i not in set(in_pair)][: quota - len(in_pair)]
        chosen += in_pair + rest
    rng.shuffle(chosen)

    # --- references: training clips of a sign from distinct signers ---
    tr_sign = train_ds.df["sign"].to_numpy()
    tr_signer = train_ds.df["participant_id"].astype(str).to_numpy()
    by_sign = {}
    for i, (s, p) in enumerate(zip(tr_sign, tr_signer)):
        by_sign.setdefault(int(s), {}).setdefault(p, []).append(i)

    clips, ref_cache = {}, {}

    def add_clip(key, ds, i):
        if key not in clips:
            clips[key] = clip_payload(ds, i)
        return key

    def tracking(j):
        """(fraction, count) of a training clip's frames with at least one hand detected."""
        _, pr = hand_presence(train_ds._load_coords(j))
        hand = pr.sum(1) > 0
        return float(hand.float().mean()), int(hand.sum())

    n_weak = 0

    def refs_for(sign):
        """One well-tracked clip from each of --refs distinct signers; weakly tracked clips
        only when a signer has nothing better, best-tracked first."""
        nonlocal n_weak
        if sign not in ref_cache:
            picks = []
            for p in by_sign.get(sign, {}):
                cand = list(by_sign[sign][p])
                rng.shuffle(cand)
                scored = [(j, *tracking(j)) for j in cand[:12]]
                good = [s for s in scored if s[1] >= args.min_ref_hand and s[2] >= args.min_ref_frames]
                picks.append((bool(good), rng.random(), (good or [max(scored, key=lambda s: (s[1], s[2]))])[0]))
            picks.sort(key=lambda t: (not t[0], t[1]))  # signers with a good clip first, random within
            chosen_refs = picks[: args.refs]
            n_weak += sum(not ok for ok, _, _ in chosen_refs)
            ref_cache[sign] = [add_clip(f"t{j}", train_ds, j) for _, _, (j, _, _) in chosen_refs]
        return ref_cache[sign]

    items = []
    for n, i in enumerate(chosen):
        t, p = int(y[i]), int(pred[i])
        true_is_a = rng.random() < 0.5
        top = np.argsort(-probs[i])[:3]
        pres = val_ds._load_coords(i)
        _, presence = hand_presence(pres)
        items.append({
            "id": n + 1,
            "query": add_clip(f"v{i}", val_ds, i),
            "a": refs_for(t if true_is_a else p),
            "b": refs_for(p if true_is_a else t),
            "trueSide": "A" if true_is_a else "B",
            "truth": idx2sign[t],
            "pred": idx2sign[p],
            "top3": [[idx2sign[int(c)], round(float(probs[i, c]), 3)] for c in top],
            "signer": signer[i],
            "frequentPair": pair(i) in freq,
            "noHand": round(float((presence.sum(1) == 0).float().mean()), 3),
            "frames": int(pres.shape[0]),
            "others": {k: idx2sign[int(v[i])] for k, v in others.items()},
            "sequenceId": str(val_ds.df["sequence_id"].iloc[i]) if "sequence_id" in val_ds.df else "",
        })

    meta = {
        "model": Path(args.checkpoint).parent.name,
        "arch": arch,
        "fold": args.val_fold,
        "accuracy": round(float(np.mean(pred == y)), 4),
        "nVal": int(len(y)),
        "nErrors": int(len(err)),
        "topPairs": [[idx2sign[a], idx2sign[b], c] for (a, b), c in Counter(pair(i) for i in err).most_common(10)],
        "seed": args.seed,
    }
    # Answers are stored per kit and indexed by item number, so the id carries a fingerprint
    # of the items: a regenerated kit with different contents never inherits old answers.
    fingerprint = hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()[:10]
    kit_id = re.sub(r"[^a-z0-9]+", "-", f"{meta['model']}-f{args.val_fold}-s{args.seed}".lower()) + "-" + fingerprint
    print(f"References: {len(clips) - len(items)} clips; {n_weak} below the tracking threshold (no better clip from that signer)")
    html = (Path(__file__).parent / "error_review_template.html").read_text()
    html = html.replace("__KIT_ID__", kit_id).replace(
        "__DATA__", json.dumps({"meta": meta, "items": items, "clips": clips}, separators=(",", ":"))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} — {len(items)} items, {len(clips)} clips, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
