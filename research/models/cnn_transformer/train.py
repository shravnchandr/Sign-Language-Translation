import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from .data.dataset import clean_subset_loader, get_data_loaders
from .data.augmentation import AdvancedAugmentation, mixup_batch
from .model.landmark_conformer import HandDominanceModule, LandmarkConformer
from .model.grl import ganin_lambda
from .model.supcon import CrossSignerSupCon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
# bf16 keeps fp32's exponent range, so activations can't overflow to inf the
# way fp16 can (Run 004 hit a NaN batch at epoch 55 under fp16) and no loss
# scaling is needed. Ampere and newer (A40/A100/3090/4090/H100) support it.
amp_dtype = (
    torch.bfloat16
    if use_amp and torch.cuda.is_bf16_supported()
    else torch.float16
)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Per-class weights handle imbalance; stored as a buffer so .to(device) works.
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


@contextmanager
def _frozen_bn_stats(model):
    """Run a train-mode forward without updating BatchNorm running statistics
    (it still normalises with batch statistics). Used for the contrastive
    pass so that enabling it changes only the objective: running statistics
    — what evaluation uses — come from the main pass alone, as without it."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom


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
    stretch_mode="batch",
    stretch_min=0.8,
    stretch_max=1.3,
    stretch_prob=0.5,
    mixup_prob=1.0,
    finger_drop_prob=0.5,
    affine_rot=15.0,
    affine_shear=0.0,
    affine_scale=0.0,
    affine_prob=0.5,
    supcon=None,
    supcon_weight=0.0,
):
    model.train()
    train_loss, correct, total = 0, 0, 0
    # Logged separately: the adversarial term (≈ ln(n_signers) at chance) jumps
    # the total as the GRL ramps on, which reads like divergence if summed.
    sign_sum, adv_sum, adv_n = 0.0, 0.0, 0
    con_sum, con_n, con_pos = 0.0, 0, 0.0
    disc_correct, disc_total = 0, 0
    optimizer.zero_grad(set_to_none=True)

    # Ganin et al. 2016 schedule: ramps from ~0 at epoch 0 to grl_lambda by mid-training.
    grl_lam = (
        ganin_lambda(epoch, total_epochs, max_lambda=grl_lambda)
        if grl_lambda > 0.0
        else 0.0
    )

    # BatchNorm running stats update during the forward pass, so a single
    # non-finite batch poisons them permanently: train mode (batch stats) keeps
    # working while eval mode (running stats) collapses to chance. GradScaler
    # protects the weights but not these buffers — snapshot and restore them.
    bn_buffers = [
        b for n, b in model.named_buffers()
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    ]
    nonfinite = 0
    canon = HandDominanceModule().to(device)  # mirrors clips before mixup

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
        # No mirror-flip: the model's HandDominanceModule mirrors every
        # right-dominant input, which would undo it on all but near-ties.

        # No extra coordinate noise here: ASLDataset.augment_sample already adds
        # σ=3e-3 noise *before* velocity is computed. A further σ=0.01 per-frame
        # term matched the whole Δ1 velocity signal (~0.01/frame) and shifted
        # joint-angle cosines by ~0.19 vs ~0.28 natural spread across hand shapes.

        # temporal_interpolation writes in-place; x is already owned (cloned above)
        x, mask = AdvancedAugmentation.temporal_interpolation(x, mask)

        if heavy_augment:
            # Time stretch. "batch": one factor for the whole batch (default,
            # Runs 001–005). "sample": an independent factor per clip.
            if stretch_mode == "sample":
                x, mask = AdvancedAugmentation.resample_per_sample(
                    x, mask, stretch_min, stretch_max, stretch_prob
                )
            elif np.random.random() > 1.0 - stretch_prob:
                x, mask = AdvancedAugmentation.time_stretch(
                    x, mask, stretch_min, stretch_max
                )

            # Rotation — batched 2×2 matmul replaces D//2 Python iterations
            sel_rot = torch.rand(B, device=x.device) > 1.0 - affine_prob
            if sel_rot.any():
                sel_idx = torch.where(sel_rot)[0]
                x[sel_idx] = AdvancedAugmentation.spatial_affine(
                    x[sel_idx], affine_rot, affine_shear, affine_scale
                )

            if finger_drop_prob > 0:
                x = AdvancedAugmentation.finger_dropout_batch(x, sample_prob=finger_drop_prob)

        # Canonicalise (mirror right-dominant clips) BEFORE mixup; the model is
        # then told not to re-mirror, since a mixture can flip dominance.
        with torch.no_grad():
            x, _ = canon(x, mask)

        x_clean, mask_clean = x, mask  # canonicalised, augmented, unmixed

        # --- Mixup ---
        y_a, y_b, lam, mixup_idx = y, None, 1.0, None
        mixed = use_mixup and np.random.random() < mixup_prob
        if mixed:
            x, y_a, y_b, lam, mask, mixup_idx = mixup_batch(x, y, mask)

        bn_snapshot = [b.detach().clone() for b in bn_buffers]
        adv_loss = None
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            use_grl = grl_lam > 0.0 and n_signers > 0
            if use_grl:
                logits, signer_logits = model(x, mask, grl_lambda=grl_lam, canonical=True)
            else:
                logits = model(x, mask, canonical=True)

            if mixed:
                sign_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(
                    logits, y_b
                )
            else:
                sign_loss = criterion(logits, y_a)

            if use_grl:
                valid = signer_ids >= 0
                if valid.any():
                    if mixed:
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
                    # grad_reverse already scales the backbone gradient by
                    # grl_lam; weighting adv_loss by it again would give the
                    # backbone −λ² and slow the discriminator by λ.
                    loss = sign_loss + adv_loss
                    # Mixup-weighted, like the adversarial loss.
                    pred_s = signer_logits[valid].argmax(dim=1)
                    hit = (pred_s == signer_ids[valid]).float().sum()
                    if mixed:
                        hit = lam * hit + (1 - lam) * (
                            pred_s == signer_ids[mixup_idx][valid]
                        ).float().sum()
                    disc_correct += hit.item()
                    disc_total += valid.sum().item()
                else:
                    loss = sign_loss
            else:
                loss = sign_loss

            con_loss = None
            if supcon is not None and supcon_weight > 0:
                # Contrastive term on the unmixed clips (a mixture belongs to
                # two signs, so it has no clean positive), second forward pass.
                base = getattr(model, "_orig_mod", model)
                with _frozen_bn_stats(base):
                    emb = model(x_clean, mask_clean, canonical=True, return_embedding=True)
                con_loss, pos_frac = supcon(base.proj_head(emb), y, signer_ids)
                loss = loss + supcon_weight * con_loss

        if not torch.isfinite(loss):
            # Undo this batch's BN stat update and skip its backward entirely.
            for buf, saved in zip(bn_buffers, bn_snapshot):
                buf.copy_(saved)
            nonfinite += 1
            continue

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
        batch_sign = sign_loss.item()
        sign_sum += batch_sign
        if adv_loss is not None:
            adv_sum += adv_loss.item()
            adv_n += 1
        if con_loss is not None:
            con_sum += con_loss.detach().item()
            con_pos += pos_frac
            con_n += 1
        # Under mixup the target is lam·y_a + (1−lam)·y_b; scoring against y_a
        # alone roughly halves reported accuracy with Beta(0.2, 0.2).
        pred = logits.argmax(dim=1)
        batch_correct = (pred == y_a).float().sum().item()
        if mixed:
            batch_correct = (
                lam * batch_correct + (1 - lam) * (pred == y_b).float().sum().item()
            )
        train_loss += batch_loss
        correct += batch_correct
        total += y_a.size(0)

        if idx % 20 == 0:
            postfix = {
                "sign": f"{batch_sign:.4f}",
                "acc": f"{batch_correct / y_a.size(0):.4f}",
            }
            if adv_loss is not None:
                postfix["adv"] = f"{adv_loss.item():.4f}"
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

    if nonfinite:
        print(
            f"  WARNING: skipped {nonfinite} non-finite batch(es) this epoch "
            f"(BN stats restored, no backward)",
            flush=True,
        )
    n_ok = max(len(data_loader) - nonfinite, 1)
    return {
        "loss": train_loss / n_ok,  # sign + adv
        "sign_loss": sign_sum / n_ok,
        "adv_loss": adv_sum / adv_n if adv_n else None,
        "con_loss": con_sum / con_n if con_n else None,
        "con_pos": con_pos / con_n if con_n else None,
        "acc": correct / max(total, 1),
        "disc_acc": disc_correct / disc_total if disc_total > 0 else None,
    }


def _train_str(stats: dict) -> str:
    """Epoch summary: sign and adversarial losses shown separately."""
    out = f"Sign Loss: {stats['sign_loss']:.4f}"
    if stats["adv_loss"] is not None:
        out += f" | Adv Loss: {stats['adv_loss']:.4f}"
    if stats.get("con_loss") is not None:
        out += f" | Con Loss: {stats['con_loss']:.4f} (pos {stats['con_pos']:.0%})"
    return out + f" | Train Acc: {stats['acc']:.4f}"


@torch.no_grad()
def predict_with_tta(model, x, mask, n_augmentations=5):
    """Average logits over the original input plus independently augmented copies.

    Accumulates a running sum rather than a list to avoid holding n_augmentations
    full batches of logits in VRAM simultaneously.
    """
    x_orig = x.clone()
    logit_sum = model(x, mask)
    for _ in range(n_augmentations - 1):
        x_aug = x_orig.clone()
        mask_aug = mask.clone()
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.gaussian_noise(x_aug, std=0.001, mask=mask_aug)
        if np.random.random() > 0.5:
            # Mild tempo jitter: resamples each clip's valid frames and rebuilds Δ1.
            x_aug, mask_aug = AdvancedAugmentation.time_stretch(x_aug, mask_aug, 0.9, 1.1)
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.spatial_rotation(x_aug, max_angle=10)
        if np.random.random() > 0.5:
            x_aug = AdvancedAugmentation.random_scale(
                x_aug, min_scale=0.95, max_scale=1.05
            )
        logit_sum = logit_sum + model(x_aug, mask_aug)
    return logit_sum / n_augmentations


def _evaluate(model, data_loader, criterion, predict, desc):
    """Return (loss, acc, per_signer) where per_signer maps each validation
    signer to its own accuracy. Accuracy varies a lot between signers, so the
    pooled number alone can't show whether a change helped everyone or one
    signer."""
    model.train(False)
    names = getattr(data_loader.dataset, "signer_names", [])
    hits = torch.zeros(len(names))
    counts = torch.zeros(len(names))
    test_loss, correct, total = 0, 0, 0
    for x, mask, y, sid in tqdm(data_loader, desc=desc, leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = predict(model, x, mask)
        test_loss += criterion(logits, y).item()
        hit = (logits.argmax(dim=1) == y).cpu()
        correct += hit.sum().item()
        total += y.size(0)
        ok = sid >= 0
        if names and ok.any():
            hits.index_add_(0, sid[ok], hit[ok].float())
            counts.index_add_(0, sid[ok], torch.ones(int(ok.sum())))
    per_signer = {n: (hits[i] / counts[i]).item() for i, n in enumerate(names) if counts[i] > 0}
    return test_loss / len(data_loader), correct / total, per_signer


@torch.no_grad()
def evaluate_epoch(model, data_loader, criterion):
    """Deterministic evaluation — no TTA — for stable model selection."""
    return _evaluate(model, data_loader, criterion, lambda m, x, k: m(x, k), "Validation")


@torch.no_grad()
def evaluate_epoch_tta(model, data_loader, criterion):
    """TTA evaluation — used for final/reporting accuracy only."""
    return _evaluate(
        model, data_loader, criterion,
        lambda m, x, k: predict_with_tta(m, x, k, n_augmentations=5), "TTA Eval",
    )


def _signer_str(per_signer: dict) -> str:
    if not per_signer:
        return ""
    vals = list(per_signer.values())
    body = " | ".join(f"{n} {a:.4f}" for n, a in per_signer.items())
    return f"  per-signer val: {body}  (spread {max(vals) - min(vals):.4f})"


def main():
    parser = argparse.ArgumentParser(
        description="Train LandmarkConformer for ASL sign recognition"
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
        "--phase2-epochs",
        type=int,
        default=0,
        help="Epochs for Phase 2 cosine warmdown (0 = skip). Off by default: it "
        "never beat the Phase 1 best in Runs 002/003 (LR jumps back to 1e-4).",
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
    parser.add_argument("--supcon-weight", type=float, default=0.0,
                        help="Weight of the cross-signer supervised contrastive loss (0 = off).")
    parser.add_argument("--supcon-temp", type=float, default=0.1, help="Contrastive temperature.")
    parser.add_argument("--supcon-queue", type=int, default=4096,
                        help="Cross-batch memory size (recent embeddings) for the contrastive loss.")
    parser.add_argument("--supcon-dim", type=int, default=128, help="Contrastive projection size.")
    parser.add_argument("--aug-rotate", type=float, default=15.0, help="Max in-plane rotation (deg).")
    parser.add_argument("--aug-shear", type=float, default=0.0, help="Max shear (0 = off).")
    parser.add_argument("--aug-scale", type=float, default=0.0,
                        help="Max per-axis (anisotropic) scale deviation (0 = off).")
    parser.add_argument("--aug-affine-prob", type=float, default=0.5,
                        help="Per-clip probability of the spatial affine (rotation/shear/scale).")
    parser.add_argument("--mixup-prob", type=float, default=1.0,
                        help="Probability a batch is mixed (1.0 = every batch, Runs 001–005; 0 = off).")
    parser.add_argument("--finger-drop-prob", type=float, default=0.5,
                        help="Per-clip probability of finger dropout (0.5 = Runs 001–005; 0 = off).")
    parser.add_argument("--zero-parts", default="",
                        help="Input ablation: comma list of {face,pose} zeroed inside the model "
                        "(train and eval). Pass the same flag when evaluating a checkpoint.")
    parser.add_argument("--no-depth", action="store_true",
                        help="Input ablation: zero all z inside the model (2D only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds Python, NumPy and torch (incl. DataLoader workers). "
                        "Not bitwise-deterministic on GPU (cudnn.benchmark).")
    parser.add_argument(
        "--train-eval-size",
        type=int,
        default=3000,
        help="Fixed un-augmented training subset evaluated each epoch in eval mode "
        "('Clean Train'), comparable to val accuracy. 0 = off.",
    )
    parser.add_argument(
        "--loss",
        choices=["focal", "ce"],
        default="focal",
        help="focal: FocalLoss with inverse-frequency class weights (Runs 001–005). "
        "ce: label-smoothed cross-entropy, no class weights (class counts 299–415).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=128,
        help="Clips longer than this are uniformly subsampled to it (5.6%% of clips "
        "exceed 128, 0.3%% exceed 256).",
    )
    parser.add_argument(
        "--hand-drop-prob",
        type=float,
        default=0.0,
        help="Per-hand probability of dropping a contiguous span of frames the way "
        "MediaPipe tracking gaps appear in the data (0 = off).",
    )
    parser.add_argument("--hand-drop-min", type=float, default=0.1,
                        help="Min dropped span, as a fraction of the clip.")
    parser.add_argument("--hand-drop-max", type=float, default=0.4,
                        help="Max dropped span, as a fraction of the clip.")
    parser.add_argument(
        "--stretch-mode",
        choices=["batch", "sample"],
        default="batch",
        help="Temporal stretch: one factor per batch (default) or per clip.",
    )
    parser.add_argument("--stretch-min", type=float, default=0.8,
                        help="Min stretch factor (<1 = faster/shorter).")
    parser.add_argument("--stretch-max", type=float, default=1.3,
                        help="Max stretch factor (>1 = slower/longer).")
    parser.add_argument("--stretch-prob", type=float, default=0.5,
                        help="Probability a batch (batch mode) or clip (sample mode) is stretched.")
    parser.add_argument(
        "--val-fold",
        type=int,
        default=None,
        help="Validate on this signer fold of GroupKFold(--n-folds) instead of the "
        "default split. Train every fold in turn for k-fold cross-validation by signer.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=7,
        help="Number of signer folds for --val-fold (7 → 3 of 21 signers per fold).",
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
    parser.add_argument(
        "--backbone-warmup-epochs",
        type=int,
        default=5,
        help="When --pretrained-backbone is set: freeze backbone for this many epochs, "
        "training only new head params (cls_token, head, geo projections, feat_fuse). "
        "After unfreezing, backbone trains at backbone-lr-ratio × head LR. "
        "0 = disabled.",
    )
    parser.add_argument(
        "--backbone-lr-ratio",
        type=float,
        default=0.1,
        help="Backbone LR as a fraction of head LR after warmup unfreezes (default 0.1 = 10× lower).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # also seeds CUDA and DataLoader worker seeds

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
        val_fold=args.val_fold,
        n_folds=args.n_folds,
        max_frames=args.max_frames,
        hand_drop=(args.hand_drop_prob, args.hand_drop_min, args.hand_drop_max),
    )

    grl_active = args.grl_lambda > 0.0 and n_signers > 0
    model = LandmarkConformer(
        num_classes=NUM_CLASSES,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        drop_path_max=args.drop_path_max,
        n_signers=n_signers if grl_active else 0,
        zero_parts=tuple(p for p in args.zero_parts.split(",") if p),
        use_depth=not args.no_depth,
        supcon_dim=args.supcon_dim if args.supcon_weight > 0 else 0,
    ).to(device)

    missing: list = []
    if args.pretrained_backbone:
        ckpt = torch.load(
            args.pretrained_backbone, map_location="cpu", weights_only=True
        )
        # strict=False still raises on shape mismatches (e.g. a backbone saved
        # before dist_proj gained the presence inputs). Drop those tensors so
        # they are re-initialised and treated as new head params below.
        own = model.state_dict()
        reshaped = [
            k for k, v in ckpt.items() if k in own and own[k].shape != v.shape
        ]
        for k in reshaped:
            del ckpt[k]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded pre-trained backbone from {args.pretrained_backbone}")
        if reshaped:
            print(f"  Re-initialised (shape changed since pre-training): {reshaped}")
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

    # Per-class weights: inverse frequency, normalised so mean weight == 1.
    # From the training split only (labels already mapped to indices).
    _train_labels = train_loader.dataset.df["sign"]
    # Reindex over every class: value_counts() omits absent labels, which would
    # shorten the vector and shift every later weight onto the wrong class.
    _counts = (
        _train_labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    )
    _weights = (1.0 / _counts.clip(lower=1).values).astype("float32")
    _weights = _weights / _weights.mean()
    class_weights = torch.tensor(_weights).to(device)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = FocalLoss(class_weights=class_weights)
    print(f"Loss        : {args.loss}  | seed {args.seed}")
    train_eval_loader = (
        clean_subset_loader(
            train_loader.dataset, args.train_eval_size, args.batch_size,
            args.num_workers, seed=args.seed,
        )
        if args.train_eval_size > 0 else None
    )

    def _clean_train_str() -> str:
        if train_eval_loader is None:
            return ""
        _, acc, _ = evaluate_epoch(model, train_eval_loader, criterion)
        return f"Clean Train: {acc:.4f} | "
    # Head params: whatever the pre-trained checkpoint did not provide (always
    # head/cls_token/signer_disc; plus any layer added after the backbone was
    # saved). Backbone params: everything that was loaded. Used to freeze the
    # backbone during warmup and set differential LR after.
    _new_params = set(missing)

    def _is_head(name: str) -> bool:
        return name.removeprefix("_orig_mod.") in _new_params

    warmup_epochs = (
        min(args.backbone_warmup_epochs, NUM_EPOCHS_PHASE1)
        if args.pretrained_backbone else 0
    )
    MAX_LR = 5e-4
    aug_kw = dict(
        stretch_mode=args.stretch_mode,
        stretch_min=args.stretch_min,
        stretch_max=args.stretch_max,
        stretch_prob=args.stretch_prob,
        mixup_prob=args.mixup_prob,
        finger_drop_prob=args.finger_drop_prob,
        affine_rot=args.aug_rotate,
        affine_shear=args.aug_shear,
        affine_scale=args.aug_scale,
        affine_prob=args.aug_affine_prob,
        supcon=(
            CrossSignerSupCon(args.supcon_dim, args.supcon_queue, args.supcon_temp).to(device)
            if args.supcon_weight > 0 else None
        ),
        supcon_weight=args.supcon_weight,
    )
    print(
        f"Spatial aug : rotate ±{args.aug_rotate}°, shear ±{args.aug_shear}, "
        f"scale ±{args.aug_scale} (p={args.aug_affine_prob})"
    )
    print(
        "Contrastive : "
        + (f"cross-signer SupCon w={args.supcon_weight}, τ={args.supcon_temp}, "
           f"queue {args.supcon_queue}, dim {args.supcon_dim}" if args.supcon_weight > 0 else "off")
    )
    print(
        f"Regularisers: mixup p={args.mixup_prob}, finger drop p={args.finger_drop_prob} | "
        f"inputs: zero_parts={args.zero_parts or 'none'}, depth={'off' if args.no_depth else 'on'}"
    )
    print(f"Max frames  : {args.max_frames}")
    print(
        "Hand dropout: "
        + (f"p={args.hand_drop_prob}, span {args.hand_drop_min}–{args.hand_drop_max} of clip"
           if args.hand_drop_prob > 0 else "off")
    )
    print(
        f"Time stretch: {args.stretch_mode} {args.stretch_min}–{args.stretch_max}× "
        f"(p={args.stretch_prob})"
    )

    # Loss scaling only matters for fp16; bf16 has fp32's range.
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"AMP dtype   : {str(amp_dtype).replace('torch.', '')}")

    def _fmt_time(seconds: float) -> str:
        s = int(seconds)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    best_acc = -float("inf")
    best_signers: dict = {}
    patience = 0
    p1_saved = False

    print("\nPhase 1: Exploration (Heavy Augmentation)")
    print("-" * 80)

    t_start_total = time.perf_counter()
    t_start_p1 = time.perf_counter()
    p1_epochs_run = 0

    # ── Backbone warmup: freeze backbone, train new heads only ────────────────
    if warmup_epochs > 0:
        print(
            f"[Warmup] Freezing backbone for {warmup_epochs} epoch(s) — "
            f"training head params only"
        )
        for name, p in model.named_parameters():
            if not _is_head(name):
                p.requires_grad_(False)

        wu_decay = [
            p for n, p in model.named_parameters()
            if _is_head(n) and p.ndim > 1 and not n.endswith(".bias")
        ]
        wu_nodecay = [
            p for n, p in model.named_parameters()
            if _is_head(n) and (p.ndim <= 1 or n.endswith(".bias"))
        ]
        wu_optimizer = optim.AdamW(
            [{"params": wu_decay, "weight_decay": 0.05},
             {"params": wu_nodecay, "weight_decay": 0.0}],
            lr=1e-4,
        )
        wu_steps = math.ceil(len(train_loader) / 4) * warmup_epochs
        wu_scheduler = optim.lr_scheduler.OneCycleLR(
            wu_optimizer, max_lr=MAX_LR, total_steps=wu_steps, pct_start=0.3,
        )

        for epoch_idx in range(warmup_epochs):
            t_epoch = time.perf_counter()
            tr = train_epoch(
                model, train_loader, wu_optimizer, criterion, scaler,
                accumulation_steps=4, use_mixup=True, heavy_augment=True,
                scheduler=wu_scheduler, epoch=epoch_idx,
                total_epochs=NUM_EPOCHS_PHASE1,
                grl_lambda=0.0, n_signers=0, **aug_kw,
            )
            v_loss, v_acc, v_signers = evaluate_epoch(model, test_loader, criterion)
            ct_str = _clean_train_str()
            epoch_secs = time.perf_counter() - t_epoch
            p1_epochs_run += 1
            print(
                f"Epoch {epoch_idx + 1:3d}/{NUM_EPOCHS_PHASE1} [warmup] | "
                f"{_train_str(tr)} | "
                f"Val Acc: {v_acc:.4f} | {ct_str}"
                f"LR: {wu_optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {_fmt_time(epoch_secs)}"
            )
            if v_signers:
                print(_signer_str(v_signers))
            if v_acc > best_acc:
                best_acc, best_signers = v_acc, v_signers
                torch.save(model.state_dict(), p1_ckpt)
                p1_saved = True
                print(f"  → Saved best P1 model ({best_acc:.4f})")

        # Unfreeze backbone; reset patience so early stopping counts from here
        for p in model.parameters():
            p.requires_grad_(True)
        patience = 0
        print(
            f"[Warmup] Backbone unfrozen — "
            f"backbone LR: {MAX_LR * args.backbone_lr_ratio:.1e}  "
            f"head LR: {MAX_LR:.1e}"
        )

    # ── Main optimizer: differential LR when backbone was pre-trained ─────────
    if args.pretrained_backbone:
        bb_max_lr = MAX_LR * args.backbone_lr_ratio
        head_decay = [
            p for n, p in model.named_parameters()
            if _is_head(n) and p.ndim > 1 and not n.endswith(".bias")
        ]
        head_nodecay = [
            p for n, p in model.named_parameters()
            if _is_head(n) and (p.ndim <= 1 or n.endswith(".bias"))
        ]
        bb_decay = [
            p for n, p in model.named_parameters()
            if not _is_head(n) and p.ndim > 1 and not n.endswith(".bias")
        ]
        bb_nodecay = [
            p for n, p in model.named_parameters()
            if not _is_head(n) and (p.ndim <= 1 or n.endswith(".bias"))
        ]
        optimizer = optim.AdamW(
            [{"params": head_decay, "weight_decay": 0.05},
             {"params": head_nodecay, "weight_decay": 0.0},
             {"params": bb_decay, "weight_decay": 0.05},
             {"params": bb_nodecay, "weight_decay": 0.0}],
            lr=1e-4,
        )
        main_max_lrs = [MAX_LR, MAX_LR, bb_max_lr, bb_max_lr]
    else:
        # No pretrained backbone — standard two-group setup (unchanged behaviour)
        decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim > 1 and not n.endswith(".bias")
        ]
        nodecay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and (p.ndim <= 1 or n.endswith(".bias"))
        ]
        optimizer = optim.AdamW(
            [{"params": decay_params, "weight_decay": 0.05},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=1e-4,
        )
        main_max_lrs = MAX_LR

    main_epochs = NUM_EPOCHS_PHASE1 - p1_epochs_run
    if main_epochs > 0:
        steps_main = math.ceil(len(train_loader) / 4) * main_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=main_max_lrs, total_steps=steps_main, pct_start=0.1,
        )
    else:
        scheduler = None

    # ── Main Phase 1 loop ─────────────────────────────────────────────────────
    for epoch_idx in range(p1_epochs_run, NUM_EPOCHS_PHASE1):
        t_epoch = time.perf_counter()
        tr = train_epoch(
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
            **aug_kw,
        )
        v_loss, v_acc, v_signers = evaluate_epoch(model, test_loader, criterion)
        ct_str = _clean_train_str()
        epoch_secs = time.perf_counter() - t_epoch
        p1_epochs_run += 1
        disc_str = f" | Disc Acc: {tr['disc_acc']:.4f}" if tr["disc_acc"] is not None else ""
        print(
            f"Epoch {epoch_idx + 1:3d}/{NUM_EPOCHS_PHASE1} | "
            f"{_train_str(tr)} | "
            f"Val Acc: {v_acc:.4f} | {ct_str}LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}" + disc_str
        )
        if v_signers:
            print(_signer_str(v_signers))
        if v_acc > best_acc:
            best_acc, best_signers = v_acc, v_signers
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
        tr = train_epoch(
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
            **aug_kw,
        )
        v_loss, v_acc, v_signers = evaluate_epoch(model, test_loader, criterion)
        ct_str = _clean_train_str()
        epoch_secs = time.perf_counter() - t_epoch
        disc_str = f" | Disc Acc: {tr['disc_acc']:.4f}" if tr["disc_acc"] is not None else ""
        print(
            f"P2 Epoch {p2_epoch + 1:3d}/{NUM_EPOCHS_PHASE2} | "
            f"{_train_str(tr)} | "
            f"Val Acc: {v_acc:.4f} | {ct_str}LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {_fmt_time(epoch_secs)}" + disc_str,
            flush=True,
        )
        if v_signers:
            print(_signer_str(v_signers))
        if v_acc > best_acc:
            best_acc, best_signers = v_acc, v_signers
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
    _, tta_acc, tta_signers = evaluate_epoch_tta(model, test_loader, criterion)
    split = "default split" if args.val_fold is None else f"fold {args.val_fold}/{args.n_folds}"
    print(f"Validation: {split}")
    print(f"Best val accuracy (deterministic): {best_acc:.4f}")
    if best_signers:
        print(_signer_str(best_signers))  # must directly follow (signer_diagnostics parses it)
    if train_eval_loader is not None:
        _, ct_acc, _ = evaluate_epoch(model, train_eval_loader, criterion)
        print(
            f"Clean train accuracy (best ckpt, {len(train_eval_loader.dataset)} clips, "
            f"no aug): {ct_acc:.4f}  → train−val gap {ct_acc - best_acc:+.4f}"
        )
    print(f"Best val accuracy (TTA):           {tta_acc:.4f}")
    if tta_signers:
        print(_signer_str(tta_signers))
    print(
        f"Phase 1 time : {_fmt_time(p1_total)} ({p1_epochs_run} epochs, avg {_fmt_time(p1_total / max(p1_epochs_run, 1))}/epoch)"
    )
    print(
        f"Phase 2 time : {_fmt_time(p2_total)} ({NUM_EPOCHS_PHASE2} epochs, avg {_fmt_time(p2_total / max(NUM_EPOCHS_PHASE2, 1))}/epoch)"
    )
    print(f"Total time   : {_fmt_time(total_time)}")


if __name__ == "__main__":
    main()
