import numpy as np
import torch
import torch.nn.functional as F
from ..config import (
    COORDS_PER_LM,
    COORD_FEAT,
    LH_START,
    N_LH,
    RH_START,
    PRESENCE_START,
    FINGER_LM_RANGES,
    FINGER_COORD_SLICES,
)


def augment_sample(
    video_coordinates: np.ndarray, noise_std: float = 3e-3, spatial_shift: float = 2e-2
) -> np.ndarray:
    video_coordinates = video_coordinates.copy()
    if np.random.random() > 0.5:
        video_coordinates += np.random.normal(0, noise_std, video_coordinates.shape)
    if np.random.random() > 0.5:
        # One rigid offset per axis, shared by every landmark. A per-column
        # offset would move each joint independently and distort hand shape
        # (±0.02 is ~40% of a finger segment).
        T, D = video_coordinates.shape
        shift = np.random.uniform(-spatial_shift, spatial_shift, COORDS_PER_LM)
        video_coordinates = (
            video_coordinates.reshape(T, -1, COORDS_PER_LM) + shift
        ).reshape(T, D)
    return video_coordinates


def hand_dropout(
    coords: torch.Tensor,
    presence: torch.Tensor,
    prob: float = 0.5,
    min_frac: float = 0.1,
    max_frac: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate MediaPipe losing a hand for a stretch of frames.

    Signer diagnostics: the fraction of frames with no hand detected ranks
    per-signer accuracy almost perfectly (Spearman -0.94 over 6 val signers),
    and well-tracked training signers rarely show such gaps. For each hand that
    is detected at all, with probability `prob`, one contiguous span covering
    U(min_frac, max_frac) of the clip is dropped exactly the way the stored data
    encodes a real gap: positions frozen at the last detection before the span
    (the first one after it, for a span at the start) and presence set to 0.

    Must run on hand_presence() output, before noise / velocity, so a dropped
    span is indistinguishable from a real one. Uses torch's RNG (seeded per
    DataLoader worker).

    coords: (T, COORD_FEAT) filled positions; presence: (T, 2) [lh, rh].
    Returns modified copies.
    """
    T = coords.shape[0]
    if prob <= 0.0 or T < 3:
        return coords, presence
    coords, presence = coords.clone(), presence.clone()
    width = N_LH * COORDS_PER_LM
    for h, start in enumerate((LH_START, RH_START)):
        if not presence[:, h].any() or torch.rand(()) >= prob:
            continue
        frac = min_frac + (max_frac - min_frac) * torch.rand(()).item()
        span = min(max(1, round(frac * T)), T - 1)  # never drop the whole clip
        s = int(torch.randint(0, T - span + 1, ()))
        e = s + span
        hold = coords[s - 1 if s > 0 else e, start : start + width]
        coords[s:e, start : start + width] = hold
        presence[s:e, h] = 0.0
    return coords, presence


class AdvancedAugmentation:
    """Advanced augmentation strategies for landmarks.

    Batch tensors are [pos | vel | presence]; geometric transforms touch only
    the coordinate channels (:PRESENCE_START), never the presence flags.

    There is no mirror-flip: HandDominanceModule mirrors every right-dominant
    input inside the model, which would undo a flip on all but near-tie samples.
    """

    @staticmethod
    def temporal_cropping(x, mask, min_ratio=0.7, max_ratio=0.95):
        B, T, D = x.shape
        crop_len = np.random.randint(int(T * min_ratio), int(T * max_ratio))
        start = np.random.randint(0, T - crop_len + 1)
        x_cropped = x[:, start : start + crop_len, :]
        mask_cropped = mask[:, start : start + crop_len]
        if crop_len < T:
            pad_len = T - crop_len
            x_padded = F.pad(x_cropped, (0, 0, 0, pad_len), value=0)
            mask_padded = F.pad(mask_cropped, (0, pad_len), value=False)
            return x_padded, mask_padded
        return x_cropped, mask_cropped

    @staticmethod
    def gaussian_noise(x, std=0.01, mask=None):
        """Position noise on valid frames, then Δ1 rebuilt from the noisy
        positions (independent noise on Δ1 would break their agreement and
        write into padded frames). Hands never detected in the clip stay
        exactly zero, as in un-augmented data."""
        x = x.clone()
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        noise = torch.randn_like(x[..., :COORD_FEAT]) * std * mask.unsqueeze(-1)
        width = N_LH * COORDS_PER_LM
        for h, hs in enumerate((LH_START, RH_START)):
            absent = x[..., PRESENCE_START + h].sum(1) == 0  # (B,)
            noise[absent, :, hs : hs + width] = 0.0
        x[..., :COORD_FEAT] += noise
        pos = x[..., :COORD_FEAT]
        vel = torch.zeros_like(pos)
        vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) * (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        x[..., COORD_FEAT:PRESENCE_START] = vel
        return x

    @staticmethod
    def temporal_interpolation(x, mask):
        """Replace isolated invalid frames with the average of their neighbours."""
        if x.shape[1] < 3:
            return x, mask
        left_valid = mask[:, :-2]
        center_inv = ~mask[:, 1:-1]
        right_valid = mask[:, 2:]
        fill_mask = left_valid & center_inv & right_valid
        fill_mask_feat = fill_mask.unsqueeze(-1).expand_as(x[:, 1:-1])
        interpolated = (x[:, :-2] + x[:, 2:]) / 2.0
        x[:, 1:-1] = torch.where(fill_mask_feat, interpolated, x[:, 1:-1])
        mask[:, 1:-1] = mask[:, 1:-1] | fill_mask
        return x, mask

    @staticmethod
    def _resample_prefix(x, mask, factors):
        """Resample each clip's valid frames by its own factor (None = keep).

        Interpolates only the valid prefix — never across the valid/padding
        boundary, which would pull the last real frames toward the origin —
        and rebuilds Δ1 from the resampled positions exactly as ASLDataset
        does, so all velocity scales agree. Presence flags are interpolated.
        Valid frames must be a mask prefix (true for collate_batch output).
        """
        B, T, D = x.shape
        lengths = mask.sum(1).tolist()
        seqs = []
        for b in range(B):
            L = int(lengths[b])
            seq = x[b, :L]
            f = factors[b]
            if f is not None and L >= 2:
                new_len = max(2, int(round(L * f)))
                if new_len != L:
                    seq = F.interpolate(
                        seq.T.unsqueeze(0), size=new_len, mode="linear", align_corners=True
                    )[0].T.clone()
                    pos = seq[:, :COORD_FEAT]
                    seq[0, COORD_FEAT:PRESENCE_START] = 0.0
                    seq[1:, COORD_FEAT:PRESENCE_START] = pos[1:] - pos[:-1]
            seqs.append(seq)
        T_new = max(len(s) for s in seqs)
        x_out = x.new_zeros(B, T_new, D)
        m_out = mask.new_zeros(B, T_new)
        for b, seq in enumerate(seqs):
            x_out[b, : len(seq)] = seq
            m_out[b, : len(seq)] = True
        return x_out, m_out

    @staticmethod
    def time_stretch(x, mask, min_stretch=0.8, max_stretch=1.3):
        """One stretch factor for the whole batch (f > 1 = slower / longer)."""
        f = float(np.random.uniform(min_stretch, max_stretch))
        return AdvancedAugmentation._resample_prefix(x, mask, [f] * x.shape[0])

    @staticmethod
    def resample_per_sample(x, mask, min_factor=0.5, max_factor=2.0, prob=0.8):
        """Per-sample temporal resampling: each selected clip gets its own
        speed factor f ~ U(min_factor, max_factor) (f > 1 = slower / longer).

        Signers differ ~2× in sign duration (hard fold-0 signers: median 45–56
        frames vs ~22 overall), which a single batch-wide 0.8–1.3× stretch
        never covers.
        """
        factors = [
            float(np.random.uniform(min_factor, max_factor))
            if np.random.random() < prob else None
            for _ in range(x.shape[0])
        ]
        return AdvancedAugmentation._resample_prefix(x, mask, factors)

    @staticmethod
    def finger_dropout(x, mask=None, dropout_prob=0.3):
        """Randomly zero out entire fingers in both hands."""
        x = x.clone()
        n_fingers = len(FINGER_LM_RANGES)
        for hand_label in ("left", "right"):
            for fi in range(n_fingers):
                if np.random.random() < dropout_prob:
                    for feat_lo, feat_hi in FINGER_COORD_SLICES[(hand_label, fi)]:
                        x[:, :, feat_lo:feat_hi] = 0.0
        return x

    @staticmethod
    def spatial_affine(x, max_angle=15, max_shear=0.0, max_scale=0.0):
        """Per-sample in-plane affine on x, y of positions and Δ1 (batched 2×2).

        A = R(θ) · Shear(s) · diag(1+a, 1+b): rotation θ ~ U(±max_angle°),
        shear s ~ U(±max_shear), per-axis scale a, b ~ U(±max_scale). Simulates
        camera angle and body-proportion differences between signers. About the
        origin (the nose), so body-relative layout is kept; a uniform scale or
        global shift is omitted (shoulder-width normalisation undoes the first,
        the second only adds a nuisance offset). Linear, so Δ1 stays equal to
        the difference of the transformed positions. z and presence untouched.
        With max_shear = max_scale = 0 this is exactly the previous rotation
        (same random draws).
        """
        B, T, _ = x.shape
        presence = x[..., PRESENCE_START:]
        x = x[..., :PRESENCE_START]
        D = PRESENCE_START
        kw = dict(dtype=x.dtype, device=x.device)
        angles = torch.tensor(np.radians(np.random.uniform(-max_angle, max_angle, B)), **kw)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        A = torch.stack(
            [torch.stack([cos_a, -sin_a], dim=-1), torch.stack([sin_a, cos_a], dim=-1)],
            dim=-2,
        )  # (B, 2, 2) rotation
        if max_shear > 0:
            sh = torch.tensor(np.random.uniform(-max_shear, max_shear, B), **kw)
            S = torch.eye(2, **kw).repeat(B, 1, 1)
            S[:, 0, 1] = sh
            A = A @ S
        if max_scale > 0:
            sc = torch.tensor(1.0 + np.random.uniform(-max_scale, max_scale, (B, 2)), **kw)
            A = A @ torch.diag_embed(sc)
        x_lm = x.reshape(B, T, D // COORDS_PER_LM, COORDS_PER_LM)
        xy = (A[:, None, None] @ x_lm[..., :2].unsqueeze(-1)).squeeze(-1)  # (B, T, K, 2)
        if COORDS_PER_LM == 2:
            return torch.cat([xy.reshape(B, T, D), presence], dim=-1)
        x_lm_out = x_lm.clone()
        x_lm_out[..., :2] = xy
        return torch.cat([x_lm_out.reshape(B, T, D), presence], dim=-1)

    @staticmethod
    def spatial_rotation(x, max_angle=15):
        """Per-sample in-plane rotation (spatial_affine without shear/scale)."""
        return AdvancedAugmentation.spatial_affine(x, max_angle)

    @staticmethod
    def finger_dropout_batch(x, sample_prob=0.5, dropout_prob=0.25):
        """Drop whole fingers by collapsing them onto their wrist.

        Runs on model input, i.e. before WristNormalization. Zeroing a finger
        there would make it −wrist after the wrist subtraction — an invented
        location that also corrupts the joint-angle geometry. Setting it (and
        its Δ1) to the wrist's makes it exactly zero in the wrist frame, the
        same as an undetected landmark.
        """
        B = x.shape[0]
        device = x.device
        x = x.clone()
        sample_gate = torch.rand(B, device=device) < sample_prob  # (B,)
        for hand_label, hs in (("left", LH_START), ("right", RH_START)):
            for fi in range(len(FINGER_LM_RANGES)):
                drop = sample_gate & (torch.rand(B, device=device) < dropout_prob)
                if not drop.any():
                    continue
                idx = drop.nonzero().flatten()
                for half, (lo, hi) in zip((0, COORD_FEAT), FINGER_COORD_SLICES[(hand_label, fi)]):
                    wrist = x[idx, :, half + hs : half + hs + COORDS_PER_LM]
                    x[idx, :, lo:hi] = wrist.repeat(1, 1, (hi - lo) // COORDS_PER_LM)
        return x

    @staticmethod
    def random_scale(x, min_scale=0.9, max_scale=1.1):
        x = x.clone()
        x[..., :PRESENCE_START] *= np.random.uniform(min_scale, max_scale)
        return x


def mixup_batch(x, y, mask, alpha=0.2):
    """Mixup on canonicalised clips, time-aligned.

    Inputs must already be mirrored to the dominant-in-LH convention (the
    training loop does this with HandDominanceModule before calling): mixing
    two raw clips can cancel the dominant hand's motion and flip which hand
    the model would mirror. Each partner is resampled to its anchor's length
    (Δ1 rebuilt), so the two signs span the same frames and the anchor's mask
    is exact — mixing different lengths otherwise leaves a tail where only
    one source contributes but the union mask calls it valid.
    """
    B, T, _ = x.shape
    index = torch.randperm(B, device=x.device)
    la, lb = [int(v) for v in mask.sum(1).tolist()], [int(v) for v in mask[index].sum(1).tolist()]
    factors = [a / b if (a != b and a >= 2 and b >= 2) else None for a, b in zip(la, lb)]
    partner, _ = AdvancedAugmentation._resample_prefix(x[index], mask[index], factors)
    partner = partner[:, :T]
    if partner.shape[1] < T:
        partner = F.pad(partner, (0, 0, 0, T - partner.shape[1]))
    # Single-frame clips can't be interpolated: a 1-frame partner is held across
    # the anchor's frames (otherwise the anchor's tail mixes with padding), and
    # a 1-frame anchor takes the partner's middle frame. Δ1 of a held frame is 0.
    src = x[index]
    for b in range(B):
        if la[b] >= 2 and lb[b] == 1:
            partner[b, : la[b]] = src[b, 0]
            partner[b, : la[b], COORD_FEAT:PRESENCE_START] = 0.0
        elif la[b] == 1 and lb[b] >= 2:
            partner[b, 0] = src[b, lb[b] // 2]
            partner[b, 0, COORD_FEAT:PRESENCE_START] = 0.0
    partner = partner * mask.unsqueeze(-1)  # anchor frames only
    lam = np.random.beta(alpha, alpha)
    return lam * x + (1 - lam) * partner, y, y[index], lam, mask, index
