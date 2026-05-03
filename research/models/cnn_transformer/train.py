import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from .data.dataset import get_data_loaders
from .data.augmentation import AdvancedAugmentation, mixup_batch
from .model.anatomical_conformer import AnatomicalConformer
from .model.grl import ganin_lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
    heavy_augment=True,
    scheduler=None,
    epoch=0,
    total_epochs=1,
    grl_lambda=0.0,
    n_signers=0,
):
    model.train()
    train_loss, correct, total = 0, 0, 0
    disc_correct, disc_total = 0, 0
    optimizer.zero_grad(set_to_none=True)

    # Ganin et al. 2016 schedule: ramps from ~0 at epoch 0 to grl_lambda by mid-training.
    grl_lam = (
        ganin_lambda(epoch, total_epochs, max_lambda=grl_lambda)
        if grl_lambda > 0.0
        else 0.0
    )

    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for idx, (x, mask, y, signer_ids) in enumerate(pbar):
        # Clone once upfront so all augmentations can write in-place without
        # extra allocations. non_blocking overlaps H2D transfer with CPU work.
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        signer_ids = signer_ids.to(device, non_blocking=True)
        B, T, D = x.shape

        # --- Per-sample augmentation (each sample gets an independent decision) ---

        if heavy_augment:
            # Flip: compute one flipped copy of the whole batch, then select per sample
            flip_sel = torch.rand(B, device=x.device) > 0.5
            if flip_sel.any():
                x_flipped = AdvancedAugmentation.random_flip(x, probability=1.0)
                x = torch.where(flip_sel[:, None, None], x_flipped, x)

        # Noise: per-sample gate; randn_like already generates independent noise per element
        noise_gate = (torch.rand(B, 1, 1, device=x.device) > 0.5).float()
        x = x + torch.randn_like(x) * 0.01 * noise_gate

        # temporal_interpolation writes in-place; x is already owned (cloned above)
        x, mask = AdvancedAugmentation.temporal_interpolation(x, mask)

        if heavy_augment:
            # Time stretch — one batch-wide interpolation replaces B serial calls
            if np.random.random() > 0.5:
                x, mask = AdvancedAugmentation.time_stretch(x, mask)

            # Rotation — batched 2×2 matmul replaces D//2 Python iterations
            sel_rot = torch.rand(B, device=x.device) > 0.5
            if sel_rot.any():
                sel_idx = torch.where(sel_rot)[0]
                x[sel_idx] = AdvancedAugmentation.spatial_rotation(
                    x[sel_idx], max_angle=15
                )

            # Finger dropout — batch mask replaces B clone+loop calls
            x = AdvancedAugmentation.finger_dropout_batch(x)

        # --- Mixup ---
        y_a, y_b, lam, mixup_idx = y, None, 1.0, None
        if use_mixup:
            x, y_a, y_b, lam, mask, mixup_idx = mixup_batch(x, y, mask)

        with autocast(device_type=device.type, enabled=use_amp):
            use_grl = grl_lam > 0.0 and n_signers > 0
            if use_grl:
                logits, signer_logits = model(x, mask, grl_lambda=grl_lam)
            else:
                logits = model(x, mask)

            if use_mixup:
                sign_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(
                    logits, y_b
                )
            else:
                sign_loss = criterion(logits, y_a)

            if use_grl:
                valid = signer_ids >= 0
                if valid.any():
                    if use_mixup and mixup_idx is not None:
                        signer_ids_b = signer_ids[mixup_idx]
                        adv_loss = lam * F.cross_entropy(
                            signer_logits[valid], signer_ids[valid]
                        ) + (1 - lam) * F.cross_entropy(
                            signer_logits[valid], signer_ids_b[valid]
                        )
                    else:
                        adv_loss = F.cross_entropy(
                            signer_logits[valid], signer_ids[valid]
                        )
                    loss = sign_loss + grl_lam * adv_loss
                    disc_correct += (
                        (signer_logits[valid].argmax(dim=1) == signer_ids[valid])
                        .sum()
                        .item()
                    )
                    disc_total += valid.sum().item()
                else:
                    loss = sign_loss
            else:
                loss = sign_loss

        scaler.scale(loss / accumulation_steps).backward()

        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        batch_loss = loss.item()
        batch_correct = (logits.argmax(dim=1) == y_a).sum().item()
        train_loss += batch_loss
        correct += batch_correct
        total += y_a.size(0)

        if idx % 20 == 0:
            postfix = {
                "loss": f"{batch_loss:.4f}",
                "acc": f"{batch_correct / y_a.size(0):.4f}",
            }
            if disc_total > 0:
                postfix["disc_acc"] = f"{disc_correct / disc_total:.4f}"
            pbar.set_postfix(postfix)

    if len(data_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    disc_acc = disc_correct / disc_total if disc_total > 0 else None
    return train_loss / len(data_loader), correct / total, disc_acc


@torch.no_grad()
def predict_with_tta(model, x, mask, n_augmentations=5):
    """Average logits over the original input plus independently augmented copies."""
    # RobustNormalization normalizes x in-place inside model(); preserve the
    # pre-normalization tensor so each TTA pass starts from the same raw input.
    x_orig = x.clone()
    predictions = [model(x, mask)]
    for _ in range(n_augmentations - 1):
        x_aug = x_orig.clone()
        mask_aug = mask.clone()
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.random_flip(x_aug, probability=1.0)
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.gaussian_noise(x_aug, std=0.005)
        if np.random.random() > 0.5:
            B_tta, T_tta, D_tta = x_aug.shape
            new_len = min(int(T_tta * np.random.uniform(0.9, 1.1)), T_tta)
            if new_len < T_tta:
                # Batch-vectorized: same stretch factor for whole batch this pass.
                x_aug = F.interpolate(
                    x_aug.permute(0, 2, 1),
                    size=new_len,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
                x_aug = F.pad(x_aug, (0, 0, 0, T_tta - new_len))
                mask_aug = (
                    F.interpolate(
                        mask_aug.float().unsqueeze(1),
                        size=new_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)
                    > 0.5
                )
                mask_aug = F.pad(mask_aug, (0, T_tta - new_len))
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
    for x, mask, y, _ in tqdm(data_loader, desc="Validation", leave=False):
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
    for x, mask, y, _ in tqdm(data_loader, desc="TTA Eval", leave=False):
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
        default="data/asl-is-lmdb",
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
        default=100,
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
    parser.add_argument(
        "--d-model", type=int, default=256, help="Conformer model width"
    )
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Conformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--drop-path-max",
        type=float,
        default=0.1,
        help="Max stochastic depth drop rate for the last Conformer block (0 = disabled)",
    )
    parser.add_argument(
        "--grl-lambda",
        type=float,
        default=0.1,
        help="Max GRL adversarial weight for signer-invariance (0 = disabled). "
        "Ramped from 0 via Ganin schedule.",
    )
    parser.add_argument(
        "--lmdb-path",
        default="data/asl-is-lmdb/is.lmdb.mdb",
        help="Path to LMDB archive. Download from shravnchandr/asl-is-lmdb or "
        "build locally with: python -m cnn_transformer.data.build_lmdb",
    )
    parser.add_argument(
        "--pretrained-backbone",
        default=None,
        help="Path to backbone_best.pth from pretrain_fingerspelling.py. "
        "Backbone weights are loaded with strict=False before training.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile(model, fullgraph=False) before training. "
        "Requires PyTorch >= 2.10 on Python 3.14. Graph breaks are allowed "
        "so the GRL custom backward does not block compilation.",
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

    print("Building data loaders...")
    train_loader, test_loader, n_signers = get_data_loaders(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        lmdb_path=args.lmdb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    grl_active = args.grl_lambda > 0.0 and n_signers > 0
    model = AnatomicalConformer(
        num_classes=NUM_CLASSES,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        drop_path_max=args.drop_path_max,
        n_signers=n_signers if grl_active else 0,
    ).to(device)

    if args.pretrained_backbone:
        ckpt = torch.load(
            args.pretrained_backbone, map_location="cpu", weights_only=True
        )
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded pre-trained backbone from {args.pretrained_backbone}")
        if missing:
            print(f"  Missing keys (expected — new head/cls_token): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    if args.compile:
        try:
            model = torch.compile(model, fullgraph=False)
            print("torch.compile enabled (fullgraph=False)")
        except Exception as e:
            print(f"torch.compile failed, falling back to eager mode: {e}")

    print(f"Num classes : {NUM_CLASSES}")
    print(
        f"Num signers : {n_signers} ({'GRL active' if grl_active else 'GRL disabled'})"
    )
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

    # Phase 1 scheduler: only created when Phase 1 will actually run.
    # OneCycleLR rejects total_steps=0, so --phase1-epochs 0 would crash.
    if NUM_EPOCHS_PHASE1 > 0:
        steps_p1 = math.ceil(len(train_loader) / 4) * NUM_EPOCHS_PHASE1
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=steps_p1,
            pct_start=0.1,
        )
    else:
        scheduler = None

    def _fmt_time(seconds: float) -> str:
        s = int(seconds)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    best_acc = -float("inf")
    patience = 0
    p1_saved = False  # guards against loading a checkpoint from a previous run

    print("\nPhase 1: Exploration (Heavy Augmentation)")
    print("-" * 80)

    t_start_total = time.perf_counter()
    t_start_p1 = time.perf_counter()
    p1_epochs_run = 0

    for epoch_idx in range(NUM_EPOCHS_PHASE1):
        t_epoch = time.perf_counter()
        t_loss, t_acc, disc_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            accumulation_steps=4,
            use_mixup=True,
            heavy_augment=True,
            scheduler=scheduler,
            epoch=epoch_idx,
            total_epochs=NUM_EPOCHS_PHASE1,
            grl_lambda=args.grl_lambda if grl_active else 0.0,
            n_signers=n_signers,
        )
        v_loss, v_acc = evaluate_epoch(model, test_loader, criterion)
        epoch_secs = time.perf_counter() - t_epoch
        p1_epochs_run += 1
        disc_str = f" | Disc Acc: {disc_acc:.4f}" if disc_acc is not None else ""
        print(
            f"Epoch {epoch_idx + 1:3d}/{NUM_EPOCHS_PHASE1} | "
            f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | "
            f"Val Acc: {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}" + disc_str
        )
        if v_acc > best_acc:
            best_acc = v_acc
            patience = 0
            torch.save(model.state_dict(), p1_ckpt)
            p1_saved = True
            print(f"  → Saved best P1 model ({best_acc:.4f})")
        else:
            patience += 1
        if patience >= MAX_PATIENCE:
            print(f"\nEarly stopping Phase 1 at epoch {epoch_idx + 1}")
            break

    p1_total = time.perf_counter() - t_start_p1
    print(
        f"\nPhase 1 complete: {p1_epochs_run} epochs | "
        f"best val acc: {best_acc:.4f} | "
        f"total time: {_fmt_time(p1_total)} | "
        f"avg per epoch: {_fmt_time(p1_total / max(p1_epochs_run, 1))}"
    )

    print("\nPhase 2: Cosine Warmdown (Heavy Augmentation Maintained)")
    print("-" * 80)

    # Load the best Phase 1 checkpoint from THIS run (not a stale file).
    if p1_saved:
        model.load_state_dict(torch.load(p1_ckpt, weights_only=True))

    # CosineAnnealing warmdown from 1e-4 → 1e-6 over Phase 2.
    # accumulation_steps matches Phase 1 so T_max counts the same unit.
    steps_p2 = math.ceil(len(train_loader) / 4) * NUM_EPOCHS_PHASE2
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"] = 1e-4
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_p2, eta_min=1e-6
    )

    final_saved = False
    t_start_p2 = time.perf_counter()

    for p2_epoch in range(NUM_EPOCHS_PHASE2):
        t_epoch = time.perf_counter()
        t_loss, t_acc, disc_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            accumulation_steps=4,
            use_mixup=True,
            heavy_augment=True,
            scheduler=scheduler_p2,
            # Continue the Ganin ramp from where Phase 1 left off so GRL stays
            # near max_lambda throughout Phase 2 rather than resetting to ~0.
            epoch=p1_epochs_run + p2_epoch,
            total_epochs=p1_epochs_run + NUM_EPOCHS_PHASE2,
            grl_lambda=args.grl_lambda if grl_active else 0.0,
            n_signers=n_signers,
        )
        v_loss, v_acc = evaluate_epoch(model, test_loader, criterion)
        epoch_secs = time.perf_counter() - t_epoch
        disc_str = f" | Disc Acc: {disc_acc:.4f}" if disc_acc is not None else ""
        print(
            f"P2 Epoch {p2_epoch + 1:3d}/{NUM_EPOCHS_PHASE2} | "
            f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | "
            f"Val Acc: {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}" + disc_str,
            flush=True,
        )
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), final_ckpt)
            final_saved = True
            print(f"  → Saved FINAL best model ({best_acc:.4f})")

    p2_total = time.perf_counter() - t_start_p2
    total_time = time.perf_counter() - t_start_total

    print("-" * 80)
    # Final TTA evaluation — load from whichever checkpoint THIS run produced.
    if final_saved:
        model.load_state_dict(torch.load(final_ckpt, weights_only=True))
    elif p1_saved:
        model.load_state_dict(torch.load(p1_ckpt, weights_only=True))
    _, tta_acc = evaluate_epoch_tta(model, test_loader, criterion)
    print(f"Best val accuracy (deterministic): {best_acc:.4f}")
    print(f"Best val accuracy (TTA):           {tta_acc:.4f}")
    print(
        f"Phase 1 time : {_fmt_time(p1_total)} ({p1_epochs_run} epochs, avg {_fmt_time(p1_total / max(p1_epochs_run, 1))}/epoch)"
    )
    print(
        f"Phase 2 time : {_fmt_time(p2_total)} ({NUM_EPOCHS_PHASE2} epochs, avg {_fmt_time(p2_total / max(NUM_EPOCHS_PHASE2, 1))}/epoch)"
    )
    print(f"Total time   : {_fmt_time(total_time)}")


if __name__ == "__main__":
    main()
