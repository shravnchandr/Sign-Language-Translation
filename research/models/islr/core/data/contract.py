"""Landmark data contract: raw MediaPipe frames → stored clip → model input. The only place it lives.

Every LMDB builder, every dataset, every evaluator and the browser fixtures go through these
functions, so training, evaluation and the app cannot drift apart (audit 2026-11-14,
docs/experiments/05_data_pipeline_audit.md).

Raw → stored (`prepare`, run once at LMDB build time):
  1. Frames without a pose are dropped (no body origin; the browser drops them too).
  2. Ghost hands: MediaPipe Holistic sometimes fits ONE physical hand into both hand slots
     (PopSign: 100 of 128 two-hand frames in a 160-clip sample; ASL Signs / ASL Citizen / WLASL
     ≈ 0). Two hands in the same frame whose 2D bounding boxes overlap by IoU >= GHOST_IOU are a
     ghost; IoU in [AMBIGUOUS_IOU, GHOST_IOU) is a ghost only if a frame within GHOST_WINDOW is a
     clear ghost, else both hands are kept and marked AMBIGUOUS. IoU of axis-aligned boxes does
     not depend on the frame's aspect ratio. Of a ghost pair, the slot that tracks this clip's
     hand on its own more often is kept (ties: the hand whose wrist is nearer its own-side pose
     wrist, aspect-corrected). Single hands are never moved between slots.
  3. Coordinates relative to the pose nose (pose is present in every kept frame).
  4. Presence = the hand was detected and kept (KEPT or AMBIGUOUS). Stored, never re-inferred.
  5. Missing landmarks hold the nearest earlier detection of their part (the first one for leading
     frames); a part never detected stays 0.
  Status per frame and hand slot records provenance: ABSENT (not detected), KEPT, GHOST (detected,
  rejected by step 2), AMBIGUOUS (detected, kept, flagged for review).

Stored → model input (`segment`, `subsample_index`, `model_input`):
  segment to the first..last frame with a hand (phrase models), keep integer positions
  (i * (T - 1)) // (M - 1) when longer than M frames, then Δ1 = frame difference (0 for the first
  frame), then [positions | Δ1 | presence]. Augmentation goes between segmenting and Δ1.
"""

import io

import numpy as np
import pandas as pd
import torch

from ..config import COORDS_PER_LM, N_LH, N_POSE, SELECTED_FACE_INDICES

CONTRACT_VERSION = "v3"  # part of the LMDB key (_cache_keys): old LMDBs fail loudly instead of mixing
ABSENT, KEPT, GHOST, AMBIGUOUS = 0, 1, 2, 3
GHOST_IOU, AMBIGUOUS_IOU, GHOST_WINDOW = 0.5, 0.2, 2

# Stored layout = config.ALL_COLUMNS: left hand, pose, right hand, selected face; x, y, z per landmark.
_PARTS = (("left_hand", N_LH), ("pose", N_POSE), ("right_hand", N_LH), ("face", len(SELECTED_FACE_INDICES)))
_FACE_POS = {lm: i for i, lm in enumerate(SELECTED_FACE_INDICES)}
_OWN_WRIST = (15, 16)  # pose landmark of the left / right wrist


def raw_arrays(df: pd.DataFrame) -> dict:
    """Competition-format rows (frame, type, landmark_index, x, y, z) → {part: (T, n, 3)} with NaN
    where a part was not detected, frames in order (`frame` holds their ids)."""
    frames = np.sort(df["frame"].unique())
    pos = {f: i for i, f in enumerate(frames)}
    out = {"frame": frames}
    for part, n in _PARTS:
        a = np.full((len(frames), n, 3), np.nan, dtype=np.float32)
        g = df[df["type"] == part]
        if part == "face":
            g = g[g["landmark_index"].isin(_FACE_POS)]
            lm = g["landmark_index"].map(_FACE_POS).to_numpy()
        else:
            g = g[g["landmark_index"] < n]
            lm = g["landmark_index"].to_numpy()
        t = g["frame"].map(pos).to_numpy()
        a[t, lm] = g[["x", "y", "z"]].to_numpy(dtype=np.float32)
        out[part] = a
    return out


def _box(h: np.ndarray) -> np.ndarray:
    return np.array([np.nanmin(h[:, 0]), np.nanmin(h[:, 1]), np.nanmax(h[:, 0]), np.nanmax(h[:, 1])])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def resolve_ghosts(raw: dict, aspect: float = 0.75) -> tuple[np.ndarray, dict]:
    """Step 2 on raw arrays (pose present in every frame) → status (T, 2) int8 and counts.
    aspect = frame width / height, only for the wrist-distance tie-break."""
    lh, rh, pose = raw["left_hand"], raw["right_hand"], raw["pose"]
    T = len(pose)
    det = np.stack([~np.isnan(lh[:, 0, 0]), ~np.isnan(rh[:, 0, 0])], 1)
    status = np.where(det, KEPT, ABSENT).astype(np.int8)
    both = np.nonzero(det.all(1))[0]
    iou = np.zeros(T)
    for t in both:
        iou[t] = _iou(_box(lh[t]), _box(rh[t]))
    strong = det.all(1) & (iou >= GHOST_IOU)
    weak = det.all(1) & (iou >= AMBIGUOUS_IOU) & ~strong
    near_strong = np.zeros(T, dtype=bool)
    for t in np.nonzero(strong)[0]:
        near_strong[max(0, t - GHOST_WINDOW): t + GHOST_WINDOW + 1] = True
    ghost = strong | (weak & near_strong)
    if ghost.any():
        alone = [int((det[:, 0] & ~det[:, 1]).sum()), int((det[:, 1] & ~det[:, 0]).sum())]
        for t in np.nonzero(ghost)[0]:
            if alone[0] != alone[1]:
                keep = int(np.argmax(alone))
            else:
                scale = np.array([aspect, 1.0])
                d = [np.linalg.norm((h[t, 0, :2] - pose[t, w, :2]) * scale) for h, w in ((lh, 15), (rh, 16))]
                keep = int(np.argmin(d))
            status[t, 1 - keep] = GHOST
    status[weak & ~ghost] = np.where(det[weak & ~ghost], AMBIGUOUS, ABSENT)
    return status, {"two_hand_frames": int(det.all(1).sum()), "ghost_frames": int(ghost.sum()),
                    "ambiguous_frames": int((weak & ~ghost).sum())}


def _hold_fill(a: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(T, ...) values; frames where present is False hold the nearest earlier present frame
    (the first present one before it); never present → 0."""
    if not present.any():
        return np.zeros_like(a)
    t = np.arange(len(a))
    src = np.maximum.accumulate(np.where(present, t, -1))
    src[src < 0] = int(np.argmax(present))
    return a[src]


def prepare(df: pd.DataFrame, aspect: float = 0.75) -> dict:
    """Steps 1–5: raw rows → {"coords": (T, COORD_FEAT) float32, "presence": (T, 2) uint8,
    "status": (T, 2) int8, "frames": kept frame ids, "stats": counts}. T can be 0 (no pose)."""
    raw = raw_arrays(df)
    posed = ~np.isnan(raw["pose"]).all((1, 2))
    stats = {"frames_raw": int(len(posed)), "frames_no_pose": int((~posed).sum())}
    raw = {k: v[posed] for k, v in raw.items()}
    status, ghost_stats = resolve_ghosts(raw, aspect)
    stats.update(ghost_stats)
    presence = np.isin(status, (KEPT, AMBIGUOUS))
    origin = raw["pose"][:, :1, :]  # nose; pose is complete whenever detected
    blocks = []
    for part, _ in _PARTS:
        a = raw[part] - origin
        if part in ("left_hand", "right_hand"):
            ok = presence[:, 0 if part == "left_hand" else 1]
            a = _hold_fill(np.where(ok[:, None, None], a, np.nan), ok)
        else:  # per landmark: MediaPipe pose / face are all-or-nothing per frame
            ok = ~np.isnan(a[:, 0, 0])
            a = _hold_fill(a, ok)
        blocks.append(np.nan_to_num(a, nan=0.0).reshape(len(a), -1))
    coords = np.concatenate(blocks, 1).astype(np.float32) if len(origin) else np.zeros((0, sum(n for _, n in _PARTS) * COORDS_PER_LM), np.float32)
    return {"coords": coords, "presence": presence.astype(np.uint8), "status": status,
            "frames": raw["frame"], "stats": stats}


def dump_clip(clip: dict) -> bytes:
    """Stored LMDB value: coords, presence, status (torch.save of a dict of tensors)."""
    buf = io.BytesIO()
    torch.save({"coords": torch.from_numpy(np.ascontiguousarray(clip["coords"])),
                "presence": torch.from_numpy(clip["presence"]).to(torch.uint8),
                "status": torch.from_numpy(clip["status"]).to(torch.int8)}, buf)
    return buf.getvalue()


def load_clip(value: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Stored LMDB value → (coords (T, COORD_FEAT) float32, presence (T, 2) float32)."""
    d = torch.load(io.BytesIO(bytes(value)), weights_only=True)
    if not isinstance(d, dict):
        raise ValueError("legacy LMDB value (coords only, presence would have to be guessed): rebuild "
                         "the LMDB with the current builders")
    return d["coords"].float(), d["presence"].float()


def segment(presence: torch.Tensor) -> slice:
    """First..last frame with any hand present (the whole clip when no hand is present)."""
    nz = torch.nonzero(presence.sum(-1) > 0).flatten()
    return slice(int(nz[0]), int(nz[-1]) + 1) if len(nz) else slice(0, presence.shape[0])


def subsample_index(T: int, max_frames: int) -> torch.Tensor:
    """Frames kept when T > max_frames: (i * (T - 1)) // (max_frames - 1) — integer arithmetic, so the
    browser computes exactly the same positions."""
    if T <= max_frames:
        return torch.arange(T)
    i = torch.arange(max_frames)
    return (i * (T - 1)) // (max_frames - 1)


def model_input(coords: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """[positions | Δ1 | presence] (T, IN_FEAT); Δ1 of the first frame is 0."""
    vel = torch.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    return torch.cat([coords, vel, presence], dim=-1)


def clip_input(coords: torch.Tensor, presence: torch.Tensor, max_frames: int, trim: bool) -> torch.Tensor:
    """Evaluation / export path for one stored clip: segment (phrase models) → subsample → Δ1."""
    if trim:
        s = segment(presence)
        coords, presence = coords[s], presence[s]
    k = subsample_index(coords.shape[0], max_frames)
    return model_input(coords[k], presence[k])
