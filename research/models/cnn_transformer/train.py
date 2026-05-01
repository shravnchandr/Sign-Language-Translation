import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from .config import INCLUDE_DEPTH
from .data.dataset import get_data_loaders
from .data.augmentation import AdvancedAugmentation, mixup_batch
from .model.anatomical_conformer import AnatomicalConformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.label_smoothing = alpha, gamma, label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        loss = self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        return loss.mean()


def train_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    scaler,
    accumulation_steps=4,
    use_mixup=True,
    scheduler=None,
):
    model.train()
    train_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for idx, (x, mask, y) in enumerate(data_loader):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        B, T, D = x.shape

        # --- Per-sample augmentation (each sample gets an independent decision) ---

        # Flip: compute one flipped copy of the whole batch, then select per sample
        flip_sel = torch.rand(B, device=x.device) > 0.5
        if flip_sel.any():
            x_flipped = AdvancedAugmentation.random_flip(x, probability=1.0)
            x = torch.where(flip_sel[:, None, None], x_flipped, x)

        # Noise: per-sample gate; randn_like already generates independent noise per element
        noise_gate = (torch.rand(B, 1, 1, device=x.device) > 0.5).float()
        x = x + torch.randn_like(x) * 0.01 * noise_gate

        # Temporal interpolation: already operates independently per (batch, time) element
        x, mask = AdvancedAugmentation.temporal_interpolation(x, mask)

        # Time stretch: loop so each sample can have a different stretch factor
        x = x.clone()  # own the tensor before in-place writes
        mask = mask.clone()
        for i in range(B):
            if np.random.random() > 0.5:
                new_len = min(int(T * np.random.uniform(0.8, 1.3)), T)
                if new_len < T:
                    xi = (
                        F.interpolate(
                            x[i].unsqueeze(0).permute(0, 2, 1),
                            size=new_len,
                            mode="linear",
                            align_corners=False,
                        )
                        .permute(0, 2, 1)
                        .squeeze(0)
                    )
                    x[i] = 0.0
                    x[i, :new_len] = xi
                    mi = F.interpolate(
                        mask[i].float().unsqueeze(0).unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze()
                    mask[i] = False
                    mask[i, :new_len] = mi > 0.5

        # Rotation: vectorized — each selected sample gets an independent angle
        sel_rot = torch.rand(B, device=x.device) > 0.5
        n_sel_rot = int(sel_rot.sum().item())
        if n_sel_rot > 0:
            sel_idx = torch.where(sel_rot)[0]
            angles_rad = torch.tensor(
                np.radians(np.random.uniform(-15, 15, n_sel_rot)),
                dtype=x.dtype,
                device=x.device,
            )
            cos_a = torch.cos(angles_rad)  # (n_sel,)
            sin_a = torch.sin(angles_rad)
            coord_stride = 3 if INCLUDE_DEPTH else 2
            for i_c in range(0, D - 1, coord_stride):
                x_c = x[sel_idx, :, i_c].clone()  # (n_sel, T)
                y_c = x[sel_idx, :, i_c + 1].clone()
                x[sel_idx, :, i_c] = x_c * cos_a.unsqueeze(1) - y_c * sin_a.unsqueeze(1)
                x[sel_idx, :, i_c + 1] = x_c * sin_a.unsqueeze(
                    1
                ) + y_c * cos_a.unsqueeze(1)

        # Finger dropout: per-sample loop (zeroing slices is cheap)
        for i in range(B):
            if np.random.random() > 0.5:
                x[i : i + 1] = AdvancedAugmentation.finger_dropout(
                    x[i : i + 1], dropout_prob=0.25
                )

        # --- Mixup ---
        y_a = y
        if use_mixup:
            x, y_a, y_b, lam, mask = mixup_batch(x, y, mask)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, mask)
            if use_mixup:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y_a)

        scaler.scale(loss / accumulation_steps).backward()

        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        train_loss += loss.item()
        correct += (logits.argmax(dim=1) == y_a).sum().item()
        total += y_a.size(0)

    if len(data_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return train_loss / len(data_loader), correct / total


@torch.no_grad()
def predict_with_tta(model, x, mask, n_augmentations=5):
    """Average logits over the original input plus independently augmented copies."""
    predictions = [model(x, mask)]
    for _ in range(n_augmentations - 1):
        # Clone so per-sample in-place ops don't touch the originals
        x_aug = x.clone()
        mask_aug = mask.clone()
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.random_flip(x_aug, probability=1.0)
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.gaussian_noise(x_aug, std=0.005)
        if np.random.random() > 0.5:
            B_tta, T_tta, _ = x_aug.shape
            for i in range(B_tta):
                new_len = min(int(T_tta * np.random.uniform(0.9, 1.1)), T_tta)
                if new_len < T_tta:
                    xi = (
                        F.interpolate(
                            x_aug[i].unsqueeze(0).permute(0, 2, 1),
                            size=new_len,
                            mode="linear",
                            align_corners=False,
                        )
                        .permute(0, 2, 1)
                        .squeeze(0)
                    )
                    x_aug[i] = 0.0
                    x_aug[i, :new_len] = xi
                    mi = F.interpolate(
                        mask_aug[i].float().unsqueeze(0).unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze()
                    mask_aug[i] = False
                    mask_aug[i, :new_len] = mi > 0.5
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.spatial_rotation(x_aug, max_angle=10)
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.random_scale(
                x_aug, min_scale=0.95, max_scale=1.05
            )
        predictions.append(model(x_aug, mask_aug))
    return torch.stack(predictions).mean(dim=0)


@torch.no_grad()
def evaluate_epoch(model, data_loader, criterion):
    """Deterministic evaluation — no TTA — for stable model selection."""
    model.train(False)
    test_loss, correct, total = 0, 0, 0
    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        test_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return test_loss / len(data_loader), correct / total


@torch.no_grad()
def evaluate_epoch_tta(model, data_loader, criterion):
    """TTA evaluation — used for final/reporting accuracy only."""
    model.train(False)
    test_loss, correct, total = 0, 0, 0
    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = predict_with_tta(model, x, mask, n_augmentations=5)
        test_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return test_loss / len(data_loader), correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train AnatomicalConformer for ASL sign recognition"
    )
    parser.add_argument(
        "--data-dir",
        default="data/Isolated_ASL_Recognition",
        help="Directory containing train.csv and sign_to_prediction_index_map.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache/cnn_transformer",
        help="Directory for per-sample .pt cache (built on first run, reused after)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/cnn_transformer",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=80,
        help="Epochs for Phase 1 (heavy augmentation)",
    )
    parser.add_argument(
        "--phase2-epochs", type=int, default=20, help="Epochs for Phase 2 (fine-tuning)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (Phase 1 only)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker processes"
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    p1_ckpt = os.path.join(args.checkpoint_dir, "best_phase1.pth")
    final_ckpt = os.path.join(args.checkpoint_dir, "best_final.pth")

    MAX_PATIENCE = args.patience
    NUM_EPOCHS_PHASE1 = args.phase1_epochs
    NUM_EPOCHS_PHASE2 = args.phase2_epochs

    sign_map_file = os.path.join(args.data_dir, "sign_to_prediction_index_map.json")
    if not os.path.exists(sign_map_file):
        raise FileNotFoundError(f"Sign map not found: {sign_map_file}")
    with open(sign_map_file) as f:
        NUM_CLASSES = len(json.load(f))

    model = AnatomicalConformer(
        num_classes=NUM_CLASSES, d_model=512, n_heads=8, n_layers=8, dropout=0.1
    ).to(device)

    print("Building data loaders (first run caches parquets as .pt files)...")
    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Num classes : {NUM_CLASSES}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss()
    # Exclude 1-D params (LayerNorm scale/bias, standalone biases) from weight decay —
    # applying L2 to these harms convergence without regularisation benefit.
    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and p.ndim > 1 and not n.endswith(".bias")
    ]
    nodecay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and (p.ndim <= 1 or n.endswith(".bias"))
    ]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.05},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=1e-4,
    )
    scaler = GradScaler(enabled=use_amp)

    # Optimizer steps per epoch = ceil(batches / accumulation_steps) because
    # train_epoch steps once more for any remainder batches (line ~149).
    # Floor division would under-count and let OneCycleLR run out of total_steps.
    steps_p1 = math.ceil(len(train_loader) / 4) * NUM_EPOCHS_PHASE1
    steps_p2 = math.ceil(len(train_loader) / 2) * NUM_EPOCHS_PHASE2
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=steps_p1 + steps_p2,
        pct_start=0.1,
    )

    best_acc = 0.0
    patience = 0
    epoch_idx = 0

    print("\nPhase 1: Exploration (Heavy Augmentation)")
    print("-" * 80)

    for epoch_idx in range(NUM_EPOCHS_PHASE1):
        t_loss, t_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            accumulation_steps=4,
            use_mixup=True,
            scheduler=scheduler,
        )
        v_loss, v_acc = evaluate_epoch(model, test_loader, criterion)
        print(
            f"Epoch {epoch_idx:3d} | Train Acc: {t_acc:.4f} | "
            f"Val Acc: {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if v_acc > best_acc:
            best_acc = v_acc
            patience = 0
            torch.save(model.state_dict(), p1_ckpt)
            print(f"  → Saved best P1 model ({best_acc:.4f})")
        else:
            patience += 1
        if patience >= MAX_PATIENCE:
            print(f"\nEarly stopping Phase 1 at epoch {epoch_idx}")
            break

    print("\nPhase 2: Fine-tuning (Gentle Augmentation)")
    print("-" * 80)

    if os.path.exists(p1_ckpt):
        model.load_state_dict(torch.load(p1_ckpt, weights_only=True))

    for p2_epoch in range(NUM_EPOCHS_PHASE2):
        t_loss, t_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            accumulation_steps=2,
            use_mixup=False,
            scheduler=scheduler,
        )
        v_loss, v_acc = evaluate_epoch(model, test_loader, criterion)
        print(
            f"P2 Epoch {p2_epoch:3d} | Train Acc: {t_acc:.4f} | "
            f"Val Acc: {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), final_ckpt)
            print(f"  → Saved FINAL best model ({best_acc:.4f})")

    print("-" * 80)
    # Final TTA evaluation on best model
    if os.path.exists(final_ckpt):
        model.load_state_dict(torch.load(final_ckpt, weights_only=True))
    elif os.path.exists(p1_ckpt):
        model.load_state_dict(torch.load(p1_ckpt, weights_only=True))
    _, tta_acc = evaluate_epoch_tta(model, test_loader, criterion)
    print(f"Best val accuracy (deterministic): {best_acc:.4f}")
    print(f"Best val accuracy (TTA):           {tta_acc:.4f}")


if __name__ == "__main__":
    main()
