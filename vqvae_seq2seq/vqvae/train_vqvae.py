"""Training script for Improved VQ-VAE."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from .config import ImprovedVQVAEConfig
from .vqvae_model import ImprovedVQVAE
from ..data.dataset import VQVAEDataset, collate_vqvae, create_dataloader
from ..data.preprocessing import LandmarkConfig, FACE_LANDMARK_SUBSETS
from ..data.signer_split import create_signer_splits


def train_epoch(
    model: ImprovedVQVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {
        "total": 0,
        "reconstruction": 0,
        "velocity_reconstruction": 0,
        "vq": 0,
        "diversity": 0,
    }
    n_batches = 0
    use_amp = device.type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(landmarks, mask)
            loss = outputs["losses"]["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses without blocking the GPU
        for key in total_losses:
            if key in outputs["losses"]:
                total_losses[key] += outputs["losses"][key].detach()
        n_batches += 1

        # .item() forces a CPU-GPU sync — only do it every 20 batches
        if n_batches % 20 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{outputs['losses']['reconstruction'].item():.4f}",
                }
            )

    return {k: (v / n_batches).item() if isinstance(v, torch.Tensor) else v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: ImprovedVQVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_losses = {
        "total": 0,
        "reconstruction": 0,
        "velocity_reconstruction": 0,
        "vq": 0,
        "diversity": 0,
    }
    codebook_usage = {}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        outputs = model(landmarks, mask)

        for key in total_losses:
            if key in outputs["losses"]:
                total_losses[key] += outputs["losses"][key].item()

        # Track codebook usage on GPU
        for name, indices in outputs["indices"].items():
            n_embed = model.quantizers.quantizers[name].num_embeddings
            usage = torch.bincount(indices.flatten(), minlength=n_embed).float()
            if name not in codebook_usage:
                codebook_usage[name] = usage
            else:
                codebook_usage[name] += usage

        n_batches += 1

    # Average losses
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Compute codebook utilization
    for name in codebook_usage:
        total = codebook_usage[name].sum()
        if total > 0:
            probs = codebook_usage[name] / total
            used = (probs > 0.001).sum().item()
            total_codes = len(probs)
            avg_losses[f"{name}_utilization"] = used / total_codes

    return avg_losses


def save_checkpoint(
    model: ImprovedVQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    losses: Dict[str, float],
    save_path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "losses": losses,
        "config": asdict(model.config),
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: ImprovedVQVAE,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Train Improved VQ-VAE")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Isolated_ASL_Recognition",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/vqvae",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory to cache preprocessed tensors (built on first run, reused after)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create config
    config = ImprovedVQVAEConfig(
        device=str(device),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Create data loaders
    print("Loading data...")
    csv_path = os.path.join(args.data_dir, "train.csv")
    splits, split_info = create_signer_splits(csv_path, args.data_dir)

    # Landmark config with compact face subset
    landmark_config = LandmarkConfig(
        include_z=True,
        face_subset=FACE_LANDMARK_SUBSETS["compact"],
    )

    train_dataset = VQVAEDataset(
        df=splits["train"],
        base_path=args.data_dir,
        config=landmark_config,
        augment=True,
        cache_dir=args.cache_dir,
    )

    val_dataset = VQVAEDataset(
        df=splits["val"],
        base_path=args.data_dir,
        config=landmark_config,
        augment=False,
        cache_dir=args.cache_dir,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_vqvae,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_vqvae,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cached = sum(1 for _ in cache_dir.rglob("*.pt")) if cache_dir.exists() else 0
        total = len(train_dataset) + len(val_dataset)
        print(f"Cache: {cached}/{total} samples ready ({cached/total*100:.1f}%) — {'loading from cache' if cached == total else 'will build cache for missing samples'}")

    # Create model
    model = ImprovedVQVAE(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.learning_rate / 100,
    )

    scaler = GradScaler(enabled=device.type == "cuda")

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Training loop — track val reconstruction (not total loss) for best-model
    # selection. The diversity term dominates total loss magnitude and is a poor
    # ranking signal for downstream tokenization quality.
    best_val_recon = float("inf")
    warmdown_epoch = int(config.max_epochs * config.codebook_reset_warmdown_ratio)
    resets_frozen = False

    for epoch in range(start_epoch, config.max_epochs):
        if not resets_frozen and epoch >= warmdown_epoch:
            model.freeze_codebook_resets()
            resets_frozen = True
            print(f"  Codebook resets disabled (warmdown from epoch {epoch}/{config.max_epochs})")

        train_losses = train_epoch(model, train_loader, optimizer, scaler, device, epoch)

        # Validate
        val_losses = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}:")
        print(
            f"  Train - Loss: {train_losses['total']:.4f}, "
            f"Recon: {train_losses['reconstruction']:.4f}, "
            f"VQ: {train_losses['vq']:.4f}"
        )
        print(
            f"  Val   - Loss: {val_losses['total']:.4f}, "
            f"Recon: {val_losses['reconstruction']:.4f}"
        )

        # Print codebook utilization
        for name in ["pose", "motion", "dynamics", "face"]:
            util_key = f"{name}_utilization"
            if util_key in val_losses:
                print(
                    f"  {name.capitalize()} codebook utilization: {val_losses[util_key]:.1%}"
                )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_losses,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"),
            )

        # Save best model based on val reconstruction loss
        val_recon = val_losses["reconstruction"]
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_losses,
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"  New best model saved! (val recon: {best_val_recon:.4f})")

    print("\nTraining complete!")
    print(f"Best val reconstruction: {best_val_recon:.4f}")


if __name__ == "__main__":
    main()
