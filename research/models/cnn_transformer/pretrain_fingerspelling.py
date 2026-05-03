"""
CTC pre-training of AnatomicalConformer on ASL Fingerspelling data.

Phase 0 of the training pipeline: forces the backbone to learn fine-grained,
signer-agnostic hand-shape representations (26 letters + digits + symbols)
before fine-tuning on the 250-class sign recognition task.

Saved artefact: backbone_best.pth — state dict of all parameters *except*
the CTC head, sign head, cls_token, and signer discriminator.  Load with
  --pretrained-backbone backbone_best.pth
when running train.py.

Usage (from project root):
  PYTHONPATH=research/models uv run python -m cnn_transformer.pretrain_fingerspelling \\
      --data-dir   data/asl-fs-lmdb \\
      --lmdb-path  data/asl-fs-lmdb/fs.lmdb.mdb \\
      --lmdb-csv   data/asl-fs-lmdb/train.csv \\
      --out-dir    checkpoints/pretrain_fs
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from .data.fingerspelling_dataset import (
    FingerspellingDataset,
    collate_ctc,
    load_char_map,
)
from .model import AnatomicalConformer

# Keys that belong to the backbone (everything that transfers to fine-tuning)
_NON_BACKBONE_PREFIXES = ("head.", "ctc_head.", "signer_disc.", "cls_token")


def _is_backbone_key(key: str) -> bool:
    return not any(key.startswith(p) for p in _NON_BACKBONE_PREFIXES)


def _fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m}m {s:02d}s"


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    char_to_idx, blank_idx = load_char_map(args.data_dir)
    vocab_size = len(char_to_idx)
    print(f"Vocab: {vocab_size} chars, blank_idx={blank_idx}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    meta = pd.read_csv(args.lmdb_csv)

    full_ds = FingerspellingDataset(
        lmdb_path=args.lmdb_path,
        csv_path=args.lmdb_csv,
        char_to_idx=char_to_idx,
        max_frames=args.max_frames,
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(meta, groups=meta["participant_id"]))
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

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
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AnatomicalConformer(
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
    print(f"Model: {n_params:,} params")

    criterion = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = math.ceil(len(train_loader) / args.accum_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(total_steps, 1),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
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
        optimizer.zero_grad(set_to_none=True)

        with tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [train]",
                  leave=False) as pbar:
            for step, (coords, mask, targets, input_lengths, target_lengths) in enumerate(pbar):
                coords = coords.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                with torch.amp.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    logits = model(coords, mask)  # (B, T, vocab+1)

                # CTC loss in FP32 — FP16 underflows with long sequences
                log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss = loss / args.accum_steps

                scaler.scale(loss).backward()
                train_loss += loss.detach() * args.accum_steps

                if (step + 1) % args.accum_steps == 0:
                    _optimizer_step(pbar)

        # Flush any remainder gradients from the last incomplete accumulation window
        remainder = len(train_loader) % args.accum_steps
        if remainder:
            _optimizer_step(tqdm([], leave=False))

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        n_val = 0
        with torch.no_grad():
            for coords, mask, targets, input_lengths, target_lengths in tqdm(
                val_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs} [val]", leave=False
            ):
                coords = coords.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                with torch.amp.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    logits = model(coords, mask)
                log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.detach()
                n_val += 1

        avg_train = (train_loss / max(len(train_loader), 1)).item()
        avg_val = (val_loss / max(n_val, 1)).item()
        epoch_secs = time.perf_counter() - t_epoch
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            backbone_sd = {
                k: v for k, v in model.state_dict().items() if _is_backbone_key(k)
            }
            torch.save(backbone_sd, out_dir / "backbone_best.pth")
            print(f"  → Saved backbone (val {avg_val:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
                break

    print("-" * 80)
    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Backbone → {out_dir}/backbone_best.pth")


def main():
    p = argparse.ArgumentParser(description="CTC pre-training on ASL Fingerspelling")
    p.add_argument(
        "--data-dir",
        required=True,
        help="data/ASL_Fingerspelling_Recognition (for char map)",
    )
    p.add_argument(
        "--lmdb-path", required=True, help="data/cache/fingerspelling/fs.lmdb"
    )
    p.add_argument(
        "--lmdb-csv", required=True, help="data/cache/fingerspelling/train.csv"
    )
    p.add_argument("--out-dir", default="checkpoints/pretrain_fs")
    # Model (must match fine-tuning config)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--drop-path-max", type=float, default=0.05)
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
