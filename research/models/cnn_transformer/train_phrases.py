"""
Continuous (multi-sign) recognition on synthetic phrases, trained with CTC.

Phrases are stitched on the fly from the ASL Signs LMDB by
SyntheticPhraseDataset (one signer per phrase, grammar from phrase_grammar),
so this needs exactly the data the isolated-sign training uses. Splits are the
same signer folds (--val-fold): validation phrases are built only from the
held-out signers' clips, and are fixed for the whole run.

The model is the usual backbone in CTC mode (per-frame logits over the 250
signs + blank). --init loads any backbone with the same architecture:
  - an isolated-sign checkpoint (best_phase1.pth) — recommended: the backbone
    already recognises single signs; CTC then learns where they are
  - an FS CTC backbone (backbone_best.pth)
  - nothing: from scratch
Only backbone keys are loaded (heads, CLS token, signer discriminator and
projection head are skipped). --n-heads must match the checkpoint: head count
is not recoverable from the weights.

Metrics on held-out signers (greedy CTC decoding):
  WER     — gloss edit distance / reference glosses (lower is better)
  exact   — share of phrases decoded exactly
  single  — exact accuracy on 1-sign phrases (comparable in spirit to
            isolated-sign accuracy, but clips are trimmed and decoded by CTC)
  WER by kind — seen: sequences training also draws (other signers' clips);
            unseen: template sequences held out of training (--holdout-frac,
            fixed hash); random: random sign sequences (almost all unseen).
            None of this replaces a test on natural continuous signing.
Best checkpoint = lowest WER. Saves best_phrases.pth (full state dict) +
phrases_meta.json, and val_examples.jsonl (reference / decoded / English).

Usage (from the project root):
  PYTHONPATH=research/models uv run python -m cnn_transformer.train_phrases \\
      --data-dir data/asl-is-lmdb --lmdb-path data/asl-is-lmdb/is.lmdb.mdb \\
      --val-fold 0 --arch graph \\
      --init results/results_fold0_graph/checkpoints/fold0_graph/best_phase1.pth \\
      --out-dir checkpoints/phrases_fold0_graph
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.augmentation import AdvancedAugmentation
from .data.dataset import BucketBatchSampler, get_data_loaders, worker_context
from .data.fingerspelling_dataset import collate_ctc
from .data.phrase_dataset import phrase_datasets
from .model import GraphConformer, LandmarkConformer
from .pretrain_fingerspelling import _fmt_time, edit_distance, greedy_decode

_SKIP_PREFIXES = ("head.", "ctc_head.", "signer_disc.", "cls_token", "proj_head.")


def build_model(args, n_classes: int):
    graph_kw = (dict(graph_hidden=args.graph_hidden, graph_layers=args.graph_layers)
                if args.arch == "graph" else {})
    return (GraphConformer if args.arch == "graph" else LandmarkConformer)(
        **graph_kw,
        num_classes=n_classes,  # unused in CTC mode
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        drop_path_max=args.drop_path_max,
        n_signers=0,
        ctc_vocab_size=n_classes,
    )


def load_backbone(model, path: str) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith(_SKIP_PREFIXES)}
    # strict=False still raises on shape mismatches (wrong d_model / graph size).
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(_SKIP_PREFIXES)]
    if unexpected or missing:
        raise SystemExit(
            f"--init {path} does not match --arch {model.__class__.__name__}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected backbone keys "
            f"(e.g. {(missing + unexpected)[:3]})"
        )
    print(f"Initialised backbone from {path} ({len(sd)} tensors)")


@torch.no_grad()
def evaluate(model, loader, val_ds, device, amp_dtype, use_amp, blank, criterion):
    model.eval()
    loss_sum, n_batches = 0.0, 0
    edits = refs_total = exact = n = single_ok = single_n = 0
    per_signer, per_kind = {}, {}
    examples = []
    pos = 0  # loader is unshuffled (sequential batches), so phrase index = running position
    for coords, mask, targets, input_lengths, target_lengths in loader:
        coords, mask = coords.to(device), mask.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(coords, mask)
        log_probs = logits.float().log_softmax(-1)
        loss = criterion(log_probs.permute(1, 0, 2), targets.to(device), input_lengths.to(device),
                         target_lengths.to(device))
        loss_sum += loss.item()
        n_batches += 1
        preds = greedy_decode(log_probs, input_lengths, blank)
        for p, r in zip(preds, torch.split(targets, target_lengths.tolist())):
            r = r.tolist()
            e = edit_distance(p, r)
            d = val_ds.describe(pos)
            for s in (per_signer.setdefault(d["signer"], [0, 0]),
                      per_kind.setdefault(d["kind"], [0, 0])):
                s[0] += e
                s[1] += len(r)
            edits += e
            refs_total += len(r)
            exact += p == r
            n += 1
            if len(r) == 1:
                single_ok += p == r
                single_n += 1
            if len(examples) < 300:
                examples.append({"signer": d["signer"], "kind": d["kind"], "ref": d["glosses"],
                                 "hyp": [val_ds.idx2sign[i] for i in p], "english": d["english"]})
            pos += 1
    return {
        "loss": loss_sum / max(n_batches, 1),
        "wer": edits / max(refs_total, 1),
        "exact": exact / max(n, 1),
        "single": single_ok / max(single_n, 1),
        "single_n": single_n,
        "per_signer": {k: v[0] / max(v[1], 1) for k, v in sorted(per_signer.items())},
        "per_kind": {k: v[0] / max(v[1], 1) for k, v in sorted(per_kind.items())},
        "examples": examples,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device}" + (f" | AMP {str(amp_dtype).replace('torch.', '')}" if use_amp else ""))

    sign2idx = json.load(open(Path(args.data_dir) / "sign_to_prediction_index_map.json"))
    idx2sign = {v: k for k, v in sign2idx.items()}
    n_classes = len(sign2idx)
    blank = n_classes

    # Clip sources: the usual signer split; lengths uncapped (max_frames is huge).
    tl, vl, _ = get_data_loaders(args.data_dir, lmdb_path=args.lmdb_path, num_workers=0,
                                 val_fold=args.val_fold, n_folds=args.n_folds, max_frames=1 << 20)
    phrase_kw = dict(max_frames=args.max_frames, random_frac=args.random_frac,
                     max_signs=args.max_signs, transition=(args.trans_min, args.trans_max),
                     min_detected=args.min_detected, holdout_frac=args.holdout_frac)
    train_ds, val_ds = phrase_datasets(
        tl.dataset, vl.dataset, idx2sign, args.train_phrases, args.val_phrases, seed=args.seed,
        tempo=(args.tempo_min, args.tempo_max),
        bone_aug=(args.bone_aug_prob, args.bone_aug_hand, args.bone_aug_bone), **phrase_kw,
    )
    if args.max_val_signers:  # quick runs: keep the phrases of the first k held-out signers
        keep = set(val_ds.signers[: args.max_val_signers])
        val_ds.specs = [s for s in val_ds.specs if s[0] in keep]
    kinds = {k: sum(s[5] == k for s in val_ds.specs) for k in ("seen", "unseen", "random")}
    print(f"Phrases: {len(train_ds):,}/epoch from {len(train_ds.signers)} training signers | "
          f"val {len(val_ds):,} fixed phrases from {sorted({s[0] for s in val_ds.specs})} "
          f"(kinds {kinds}; 1-sign phrases count as seen)")

    worker_kw = dict(num_workers=args.num_workers, collate_fn=collate_ctc,
                     multiprocessing_context=worker_context(args.num_workers),
                     pin_memory=device.type == "cuda")
    # Validation in fixed order so evaluate() can map results back to phrases.
    val_order = np.argsort(val_ds.lengths(), kind="stable")
    val_ds.specs = [val_ds.specs[i] for i in val_order]
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **worker_kw)

    model = build_model(args, n_classes).to(device)
    if args.init:
        load_backbone(model, args.init)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params | arch {args.arch} | "
          f"{args.n_heads} heads | seed {args.seed}")

    criterion = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(BucketBatchSampler(train_ds.lengths(), args.batch_size))
                                / args.accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(steps_per_epoch * args.epochs, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    bn_buffers = [b for n, b in model.named_buffers()
                  if n.endswith(("running_mean", "running_var", "num_batches_tracked"))]

    def optimizer_step():
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() == scale_before and scheduler.last_epoch < scheduler.total_steps - 1:
            scheduler.step()

    if args.eval_init:
        r = evaluate(model, val_loader, val_ds, device, amp_dtype, use_amp, blank, criterion)
        print(f"Before training | Val WER {r['wer']:.4f} | exact {r['exact']:.4f} | "
              f"single {r['single']:.4f}")

    best_wer, patience = float("inf"), 0
    print("\nPhrase CTC training\n" + "-" * 80)
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_ds.set_epoch(epoch)  # new phrases every epoch
        train_loader = DataLoader(
            train_ds, batch_sampler=BucketBatchSampler(train_ds.lengths(), args.batch_size),
            **worker_kw)
        model.train()
        loss_sum, n_ok, nonfinite = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        n_steps = len(train_loader)
        for step, (coords, mask, targets, _, target_lengths) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1:3d}/{args.epochs}", leave=False)):
            coords, mask = coords.to(device), mask.to(device)
            sel = torch.rand(coords.shape[0], device=device) < args.aug_affine_prob
            if sel.any():
                i = torch.where(sel)[0]
                coords[i] = AdvancedAugmentation.spatial_affine(coords[i], args.aug_rotate)
            input_lengths = mask.sum(1).long()
            snap = [b.clone() for b in bn_buffers]
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(coords, mask)
            log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)
            loss = criterion(log_probs, targets.to(device), input_lengths, target_lengths.to(device))
            if torch.isfinite(loss):
                scaler.scale(loss / args.accum_steps).backward()
                loss_sum += loss.item()
                n_ok += 1
            else:  # a NaN forward already touched BatchNorm stats — restore them
                with torch.no_grad():
                    for b, s in zip(bn_buffers, snap):
                        b.copy_(s)
                nonfinite += 1
            if (step + 1) % args.accum_steps == 0 or step + 1 == n_steps:
                optimizer_step()

        r = evaluate(model, val_loader, val_ds, device, amp_dtype, use_amp, blank, criterion)
        signers = " ".join(f"{k}:{v:.3f}" for k, v in r["per_signer"].items())
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | Train Loss {loss_sum / max(n_ok, 1):.4f} | "
              f"Val Loss {r['loss']:.4f} | Val WER {r['wer']:.4f} | exact {r['exact']:.4f} | "
              f"single {r['single']:.4f} (n={r['single_n']}) | LR {scheduler.get_last_lr()[0]:.2e} | "
              f"Time {_fmt_time(time.perf_counter() - t0)}"
              + (f" | non-finite skipped {nonfinite}" if nonfinite else ""))
        kinds = " ".join(f"{k}:{v:.3f}" for k, v in r["per_kind"].items())
        print(f"  per-signer WER: {signers} | by kind: {kinds}")
        if r["wer"] < best_wer:
            best_wer, patience = r["wer"], 0
            torch.save(model.state_dict(), out_dir / "best_phrases.pth")
            meta = {"arch": args.arch, "d_model": args.d_model, "n_layers": args.n_layers,
                    "n_heads": args.n_heads, "vocab": "sign_to_prediction_index_map.json",
                    "blank": blank, "val_fold": args.val_fold, "epoch": epoch + 1,
                    "val_wer": r["wer"], "val_exact": r["exact"], "init": args.init}
            if args.arch == "graph":
                meta.update(graph_hidden=args.graph_hidden, graph_layers=args.graph_layers)
            (out_dir / "phrases_meta.json").write_text(json.dumps(meta, indent=2))
            with open(out_dir / "val_examples.jsonl", "w") as f:
                for ex in r["examples"]:
                    f.write(json.dumps(ex) + "\n")
            print(f"  → saved best (val WER {best_wer:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (patience {args.patience})")
                break
    print("-" * 80 + f"\nBest val WER {best_wer:.4f} → {out_dir}/best_phrases.pth")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/asl-is-lmdb")
    p.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb")
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=7)
    p.add_argument("--out-dir", default="checkpoints/phrases")
    p.add_argument("--init", default="", help="backbone checkpoint (best_phase1.pth / backbone_best.pth)")
    p.add_argument("--eval-init", action="store_true", help="evaluate once before training")
    # Model (must match --init)
    p.add_argument("--arch", choices=["conformer", "graph"], default="graph")
    p.add_argument("--graph-hidden", type=int, default=64)
    p.add_argument("--graph-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-max", type=float, default=0.1)
    # Phrases
    p.add_argument("--train-phrases", type=int, default=40000, help="per epoch (redrawn each epoch)")
    p.add_argument("--val-phrases", type=int, default=3000)
    p.add_argument("--max-val-signers", type=int, default=0, help="0 = all held-out signers")
    p.add_argument("--random-frac", type=float, default=0.3, help="random sign sequences vs templates")
    p.add_argument("--max-signs", type=int, default=5)
    p.add_argument("--trans-min", type=int, default=3)
    p.add_argument("--trans-max", type=int, default=8)
    p.add_argument("--tempo-min", type=float, default=0.8)
    p.add_argument("--tempo-max", type=float, default=1.25)
    p.add_argument("--min-detected", type=int, default=4)
    p.add_argument("--holdout-frac", type=float, default=0.1,
                   help="multi-sign sequences never trained on (validation: 'unseen')")
    p.add_argument("--max-frames", type=int, default=384)
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aug-rotate", type=float, default=15.0)
    p.add_argument("--aug-affine-prob", type=float, default=0.5)
    p.add_argument("--bone-aug-prob", type=float, default=0.0)
    p.add_argument("--bone-aug-hand", type=float, default=0.15)
    p.add_argument("--bone-aug-bone", type=float, default=0.1)
    train(p.parse_args())


if __name__ == "__main__":
    main()
