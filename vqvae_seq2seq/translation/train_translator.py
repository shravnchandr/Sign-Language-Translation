"""Training script for Sign Language Translator."""

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
from torch.optim.lr_scheduler import LambdaLR

from .config import TranslationConfig
from .translator_model import SignTranslator
from ..data.dataset import (
    TranslationDataset,
    TokenizedTranslationDataset,
    collate_translation,
    collate_tokenized,
    create_dataloader,
)
from ..data.preprocessing import LandmarkConfig, FACE_LANDMARK_SUBSETS
from ..data.signer_split import create_signer_splits
from ..data.vocabulary import GlossVocabulary
from ..vqvae.config import ImprovedVQVAEConfig
from ..vqvae.vqvae_model import ImprovedVQVAE


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Create learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training loop for Sign Translator."""

    def __init__(
        self,
        model: SignTranslator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocabulary: GlossVocabulary,
        config: TranslationConfig,
        device: torch.device,
        output_dir: str,
        vqvae: Optional[ImprovedVQVAE] = None,
    ):
        self.model = model.to(device)
        # vqvae is None when using pre-tokenized data (TokenizedTranslationDataset)
        if vqvae is not None:
            self.vqvae = vqvae.to(device)
            self.vqvae.train(False)
        else:
            self.vqvae = None

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.device = device
        self.output_dir = output_dir

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def _get_token_indices(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Return token indices — from pre-tokenized batch or by running VQ-VAE."""
        factor_keys = {"pose", "motion", "dynamics", "face"}
        if factor_keys.issubset(batch.keys()):
            # Pre-tokenized: indices already in batch, just move to device
            return {k: batch[k].to(self.device) for k in factor_keys if k in batch}

        # Fallback: run frozen VQ-VAE on raw landmarks
        with torch.no_grad():
            landmarks = batch["landmarks"].to(self.device)
            mask = batch["mask"].to(self.device)
            return self.vqvae.tokenize(landmarks, mask)

    def _prepare_targets(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare target sequences with BOS token."""
        B = labels.shape[0]
        device = labels.device

        # For isolated sign recognition, target is just the single sign
        # Wrap with BOS and EOS
        targets = torch.full(
            (B, 3),  # BOS, label, EOS
            self.config.pad_idx,
            dtype=torch.long,
            device=device,
        )
        targets[:, 0] = self.config.bos_idx
        targets[:, 1] = labels
        targets[:, 2] = self.config.eos_idx

        return targets

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {"total": 0, "ctc": 0, "attention": 0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            token_indices = self._get_token_indices(batch)

            labels = batch["labels"].to(self.device)
            targets = self._prepare_targets(labels)

            if "lengths" in batch:
                # Pre-tokenized: lengths are already in chunk units
                encoder_lengths = batch["lengths"].to(self.device)
            else:
                mask = batch["mask"].to(self.device)
                encoder_lengths = ((~mask).sum(dim=1) // 8).clamp(min=1)

            target_lengths = torch.ones(
                labels.shape[0], device=self.device
            )  # Single token targets

            # Forward pass
            self.optimizer.zero_grad()
            losses = self.model(
                token_indices,
                targets,
                encoder_mask=None,  # Tokens don't need mask after VQ-VAE
                encoder_lengths=encoder_lengths,
                target_lengths=target_lengths,
            )

            # Backward
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_losses = {"total": 0, "ctc": 0, "attention": 0}
        correct = 0
        total = 0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            token_indices = self._get_token_indices(batch)

            labels = batch["labels"].to(self.device)
            targets = self._prepare_targets(labels)

            if "lengths" in batch:
                encoder_lengths = batch["lengths"].to(self.device)
            else:
                mask = batch["mask"].to(self.device)
                encoder_lengths = ((~mask).sum(dim=1) // 8).clamp(min=1)
            target_lengths = torch.ones(labels.shape[0], device=self.device)

            # Forward
            losses = self.model(
                token_indices,
                targets,
                encoder_lengths=encoder_lengths,
                target_lengths=target_lengths,
            )

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()

            predictions = self.model.translate(token_indices, use_beam_search=False)

            for pred, label in zip(predictions, labels.tolist()):
                if len(pred) > 0 and pred[0] == label:
                    correct += 1
                total += 1

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses["accuracy"] = correct / total if total > 0 else 0

        return avg_losses

    def train(self):
        """Full training loop."""
        for epoch in range(self.config.max_epochs):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_losses['total']:.4f}")
            print(
                f"  Val   - Loss: {val_metrics['total']:.4f}, Acc: {val_metrics['accuracy']:.1%}"
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics)

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, val_metrics, best=True)
                print(f"  New best accuracy: {self.best_val_acc:.1%}")

        print(f"\nTraining complete! Best accuracy: {self.best_val_acc:.1%}")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best: bool = False,
    ):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
        }

        if best:
            path = os.path.join(self.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Translator")
    parser.add_argument("--data-dir", type=str, default="data/Isolated_ASL_Recognition")
    parser.add_argument(
        "--vqvae-checkpoint",
        type=str,
        default=None,
        help="Path to trained VQ-VAE checkpoint (not needed when --token-dir is set)",
    )
    parser.add_argument(
        "--token-dir",
        type=str,
        default=None,
        help="Directory of pre-tokenized .pt files (from precompute_tokens.py). "
             "When provided, VQ-VAE is not loaded and not kept in memory.",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/translator")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.token_dir is None and args.vqvae_checkpoint is None:
        parser.error("Provide either --token-dir (fast) or --vqvae-checkpoint (slow).")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    sign_map_path = os.path.join(args.data_dir, "sign_to_prediction_index_map.json")
    with open(sign_map_path) as f:
        sign_to_idx = json.load(f)

    vocabulary = GlossVocabulary.from_sign_to_prediction_map(sign_map_path)
    print(f"Vocabulary size: {len(vocabulary)}")

    csv_path = os.path.join(args.data_dir, "train.csv")
    splits, _ = create_signer_splits(csv_path, args.data_dir)

    # ── Data loading ─────────────────────────────────────────────────────────
    vqvae = None

    if args.token_dir is not None:
        print(f"Using pre-tokenized data from {args.token_dir}")
        train_dataset = TokenizedTranslationDataset(
            df=splits["train"],
            token_dir=args.token_dir,
            sign_to_idx=sign_to_idx,
        )
        val_dataset = TokenizedTranslationDataset(
            df=splits["val"],
            token_dir=args.token_dir,
            sign_to_idx=sign_to_idx,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_tokenized,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_tokenized,
        )
    else:
        print(f"Loading VQ-VAE from {args.vqvae_checkpoint}...")
        vqvae_checkpoint = torch.load(
            args.vqvae_checkpoint, map_location=device, weights_only=True
        )
        vqvae_config = ImprovedVQVAEConfig(**vqvae_checkpoint["config"])
        vqvae = ImprovedVQVAE(vqvae_config)
        vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
        vqvae = vqvae.to(device)
        vqvae.train(False)

        landmark_config = LandmarkConfig(
            include_z=True,
            face_subset=FACE_LANDMARK_SUBSETS["compact"],
        )
        train_dataset = TranslationDataset(
            df=splits["train"],
            base_path=args.data_dir,
            sign_to_idx=sign_to_idx,
            config=landmark_config,
            augment=True,
        )
        val_dataset = TranslationDataset(
            df=splits["val"],
            base_path=args.data_dir,
            sign_to_idx=sign_to_idx,
            config=landmark_config,
            augment=False,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_translation,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_translation,
        )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Config ───────────────────────────────────────────────────────────────
    config = TranslationConfig(
        vocab_size=len(vocabulary),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        device=str(device),
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SignTranslator(config)

    if vqvae is not None:
        codebooks = vqvae.get_codebook_embeddings()
        model.init_from_vqvae(codebooks)
    elif args.token_dir is not None:
        # Load codebook embeddings from the meta checkpoint saved during precompute
        import json as _json
        meta_path = Path(args.token_dir) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            vqvae_ckpt_path = meta.get("vqvae_checkpoint")
            if vqvae_ckpt_path and Path(vqvae_ckpt_path).exists():
                print(f"Initializing embeddings from {vqvae_ckpt_path}")
                tmp_ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=True)
                tmp_cfg = ImprovedVQVAEConfig(**tmp_ckpt["config"])
                tmp_model = ImprovedVQVAE(tmp_cfg)
                tmp_model.load_state_dict(tmp_ckpt["model_state_dict"])
                codebooks = tmp_model.get_codebook_embeddings()
                model.init_from_vqvae(codebooks)
                del tmp_model, tmp_ckpt

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        config=config,
        device=device,
        output_dir=args.output_dir,
        vqvae=vqvae,
    )
    trainer.train()


if __name__ == "__main__":
    main()
