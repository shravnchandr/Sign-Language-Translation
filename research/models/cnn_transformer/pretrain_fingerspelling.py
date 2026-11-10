"""
CTC pre-training of LandmarkConformer / GraphConformer on ASL Fingerspelling.

Phase 0 of the training pipeline: forces the backbone to learn fine-grained,
signer-agnostic hand-shape representations (26 letters + digits + symbols)
before fine-tuning on the 250-class sign recognition task. Fingerspelling has
94 signers vs 21 in ASL Signs, so this is the cheapest way to show the
backbone more signers.

Saved artefacts:
  backbone_best.pth   — state dict of all parameters *except* the CTC head,
                        sign head, cls_token, and signer discriminator.
  backbone_meta.json  — architecture of that backbone; train.py refuses a
                        backbone whose architecture differs from the fine-tune.
Load with
  --pretrained-backbone backbone_best.pth
when running train.py (same --arch / --d-model / --n-layers / --n-heads).

Best checkpoint = lowest greedy-decoded character error rate (CER) on held-out
signers; CTC loss alone cannot show whether letters are actually recognised.

Usage (from project root):
  PYTHONPATH=research/models uv run python -m cnn_transformer.pretrain_fingerspelling \\
      --data-dir   data/asl-fs-lmdb \\
      --lmdb-path  data/asl-fs-lmdb/fs.lmdb.mdb \\
      --lmdb-csv   data/asl-fs-lmdb/train.csv \\
      --out-dir    checkpoints/pretrain_fs
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from .data.augmentation import AdvancedAugmentation
from .data.fingerspelling_dataset import (
    FingerspellingDataset,
    collate_ctc,
    load_char_map,
)
from .model import GraphConformer, LandmarkConformer

# Keys that belong to the backbone (everything that transfers to fine-tuning)
_NON_BACKBONE_PREFIXES = ("head.", "ctc_head.", "signer_disc.", "cls_token")


def _is_backbone_key(key: str) -> bool:
    return not any(key.startswith(p) for p in _NON_BACKBONE_PREFIXES)


def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m}m {s:02d}s"


def backbone_meta(args) -> dict:
    """Architecture fields that must match between pre-training and fine-tuning."""
    meta = {"arch": args.arch, "d_model": args.d_model,
            "n_layers": args.n_layers, "n_heads": args.n_heads}
    if args.arch == "graph":
        meta.update(graph_hidden=args.graph_hidden, graph_layers=args.graph_layers)
    return meta


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int) -> list:
    """(B, T, C) → per-sample char-index lists: argmax, collapse repeats, drop blanks."""
    best = log_probs.argmax(-1).cpu()
    out = []
    for b, L in enumerate(lengths.tolist()):
        seq, prev = [], None
        for t in best[b, :L].tolist():
            if t != prev and t != blank:
                seq.append(t)
            prev = t
        out.append(seq)
    return out


def edit_distance(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # also seeds CUDA and DataLoader worker seeds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    # As train.py: bf16 on Ampere+ (no loss scaling, no fp16 overflow); fp16 +
    # GradScaler otherwise.
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    print(f"Device: {device}" + (f" | AMP {str(amp_dtype).replace('torch.', '')}" if use_amp else ""))

    char_to_idx, blank_idx = load_char_map(args.data_dir)
    vocab_size = len(char_to_idx)
    print(f"Vocab: {vocab_size} chars, blank_idx={blank_idx}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    meta = pd.read_csv(args.lmdb_csv)
    ds_kw = dict(lmdb_path=args.lmdb_path, csv_path=args.lmdb_csv,
                 char_to_idx=char_to_idx, max_frames=args.max_frames)
    train_full = FingerspellingDataset(
        **ds_kw, augment=args.augment,
        bone_aug=(args.bone_aug_prob, args.bone_aug_hand, args.bone_aug_bone),
    )
    val_full = FingerspellingDataset(**ds_kw)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(meta, groups=meta["participant_id"]))
    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(val_full, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ctc,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ctc,
        pin_memory=(device.type == "cuda"),
    )
    n_train_signers = meta.iloc[train_idx]["participant_id"].nunique()
    n_val_signers = meta.iloc[val_idx]["participant_id"].nunique()
    print(f"Train: {len(train_ds):,} ({n_train_signers} signers)  "
          f"Val: {len(val_ds):,} ({n_val_signers} held-out signers)")
    print(
        "Augment: "
        + (f"noise/shift, affine rot ±{args.aug_rotate}° shear ±{args.aug_shear} "
           f"scale ±{args.aug_scale} (p={args.aug_affine_prob}), stretch "
           f"{args.stretch_min}–{args.stretch_max}× (p={args.stretch_prob}), bone "
           + (f"p={args.bone_aug_prob}" if args.bone_aug_prob > 0 else "off")
           if args.augment else "off")
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    graph_kw = (
        dict(graph_hidden=args.graph_hidden, graph_layers=args.graph_layers)
        if args.arch == "graph" else {}
    )
    model = (GraphConformer if args.arch == "graph" else LandmarkConformer)(
        **graph_kw,
        num_classes=250,  # unused in CTC mode
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        drop_path_max=args.drop_path_max,
        n_signers=0,
        ctc_vocab_size=vocab_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | arch {args.arch} | seed {args.seed}")

    criterion = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = math.ceil(len(train_loader) / args.accum_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(total_steps, 1),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # A NaN forward updates BatchNorm running stats before the loss is seen,
    # silently breaking eval mode (Run 004) — snapshot and restore them.
    bn_buffers = [
        b for n, b in model.named_buffers()
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    ]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_cer = float("inf")
    patience = 0

    print("\nCTC Pre-training")
    print("-" * 80)

    def _optimizer_step(pbar: tqdm) -> None:
        """Unscale → clip → step → update. Only advances scheduler when the
        optimizer actually updated weights (GradScaler may skip on NaN/Inf)."""
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() == scale_before:
            scheduler.step()
        pbar.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.1e}", refresh=False)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        n_train = 0
        nonfinite = 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [train]", leave=False
        ) as pbar:
            for step, (
                coords,
                mask,
                targets,
                input_lengths,
                target_lengths,
            ) in enumerate(pbar):
                coords = coords.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                if args.augment:
                    # Length-changing first; the CTC input lengths follow the mask.
                    # Per-clip tempo (Δ1 rebuilt). No mixup, hand or finger
                    # dropout: each would change which letters the clip shows.
                    if args.stretch_prob > 0:
                        coords, mask = AdvancedAugmentation.resample_per_sample(
                            coords, mask, args.stretch_min, args.stretch_max, args.stretch_prob
                        )
                    sel = torch.rand(coords.shape[0], device=device) < args.aug_affine_prob
                    if sel.any():
                        idx = torch.where(sel)[0]
                        coords[idx] = AdvancedAugmentation.spatial_affine(
                            coords[idx], args.aug_rotate, args.aug_shear, args.aug_scale
                        )
                input_lengths = mask.sum(1).long()

                bn_snapshot = [b.clone() for b in bn_buffers]
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    logits = model(coords, mask)  # (B, T, vocab+1)
                # CTC loss in FP32 — FP16 underflows with long sequences
                log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    scaler.scale(loss / args.accum_steps).backward()
                    train_loss += loss.detach()
                    n_train += 1
                else:  # skip the backward; the window still steps on schedule
                    with torch.no_grad():
                        for b, s in zip(bn_buffers, bn_snapshot):
                            b.copy_(s)
                    nonfinite += 1

                if (step + 1) % args.accum_steps == 0:
                    _optimizer_step(pbar)

        # Flush any remainder gradients from the last incomplete accumulation window
        remainder = len(train_loader) % args.accum_steps
        if remainder:
            _optimizer_step(tqdm([], leave=False))

        # ── Validation: CTC loss + greedy CER on held-out signers ────────────
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        n_val = 0
        edits = chars = 0
        with torch.no_grad():
            for coords, mask, targets, input_lengths, target_lengths in tqdm(
                val_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [val]", leave=False
            ):
                coords = coords.to(device)
                mask = mask.to(device)
                input_lengths = input_lengths.to(device)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    logits = model(coords, mask)
                log_probs = logits.float().log_softmax(-1)
                loss = criterion(
                    log_probs.permute(1, 0, 2), targets.to(device), input_lengths,
                    target_lengths.to(device),
                )
                val_loss += loss.detach()
                n_val += 1
                preds = greedy_decode(log_probs, input_lengths, blank_idx)
                refs = torch.split(targets, target_lengths.tolist())
                for p, r in zip(preds, refs):
                    edits += edit_distance(p, r.tolist())
                    chars += len(r)

        avg_train = (train_loss / max(n_train, 1)).item()
        avg_val = (val_loss / max(n_val, 1)).item()
        cer = edits / max(chars, 1)
        epoch_secs = time.perf_counter() - t_epoch
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val CER: {cer:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}"
            + (f" | non-finite batches skipped: {nonfinite}" if nonfinite else "")
        )

        if cer < best_cer:
            best_cer = cer
            patience = 0
            backbone_sd = {
                k: v for k, v in model.state_dict().items() if _is_backbone_key(k)
            }
            torch.save(backbone_sd, out_dir / "backbone_best.pth")
            (out_dir / "backbone_meta.json").write_text(
                json.dumps({**backbone_meta(args), "epoch": epoch + 1, "val_cer": cer}, indent=2)
            )
            print(f"  → Saved backbone (val CER {cer:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
                break

    print("-" * 80)
    print(f"\nPre-training complete. Best val CER: {best_cer:.4f}")
    print(f"Backbone → {out_dir}/backbone_best.pth")


def main():
    p = argparse.ArgumentParser(description="CTC pre-training on ASL Fingerspelling")
    p.add_argument(
        "--data-dir",
        required=True,
        help="data/asl-fs-lmdb (for character_to_prediction_index.json)",
    )
    p.add_argument(
        "--lmdb-path", required=True, help="data/asl-fs-lmdb/fs.lmdb.mdb"
    )
    p.add_argument(
        "--lmdb-csv", required=True, help="data/asl-fs-lmdb/train.csv"
    )
    p.add_argument("--out-dir", default="checkpoints/pretrain_fs")
    # Model (must match fine-tuning config; recorded in backbone_meta.json)
    p.add_argument("--arch", choices=["conformer", "graph"], default="conformer")
    p.add_argument("--graph-hidden", type=int, default=64)
    p.add_argument("--graph-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-max", type=float, default=0.05)
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience on val CER"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Augmentation (train split only). Defaults = train.py's defaults, except
    # stretch: per clip and milder (letters are short; CTC needs T ≥ target).
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable all augmentation (previous behaviour).")
    p.add_argument("--aug-rotate", type=float, default=15.0)
    p.add_argument("--aug-shear", type=float, default=0.0)
    p.add_argument("--aug-scale", type=float, default=0.0)
    p.add_argument("--aug-affine-prob", type=float, default=0.5)
    p.add_argument("--stretch-min", type=float, default=0.8)
    p.add_argument("--stretch-max", type=float, default=1.25)
    p.add_argument("--stretch-prob", type=float, default=0.5)
    p.add_argument("--bone-aug-prob", type=float, default=0.0)
    p.add_argument("--bone-aug-hand", type=float, default=0.15)
    p.add_argument("--bone-aug-bone", type=float, default=0.1)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
