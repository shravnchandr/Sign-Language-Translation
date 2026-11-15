"""Landmark data contract: raw MediaPipe frames → stored clip → model input. The only place it lives.

Every LMDB builder, every dataset, every evaluator and the browser fixtures go through these
functions, so training, evaluation and the app cannot drift apart (audit 2026-11-14,
docs/experiments/05_data_pipeline_audit.md).

Raw → stored (`prepare`, run once at LMDB build time):
  0. Raw rows are checked (check_raw): known part types, integral frame ids, landmark indices
     inside their part, no repeated (frame, type, landmark_index) record. With frame_step k > 1
     (video recorded above ~30 fps: k = frame_step_for(fps) = round(fps / TARGET_FPS), recorded per
     clip), only the frames at positions 0, k, 2k, … of the sorted frame ids are kept — what a
     ~30 fps camera samples; everything below runs on those frames.
  1. Frames without a pose are dropped (no body origin; the browser drops them too).
  2. Copied hands (contract v4, chosen on visually labelled frames — docs/experiments/
     05_data_pipeline_audit.md, "Ghost rule v4"): MediaPipe Holistic sometimes fits ONE physical
     hand into both hand slots. In a frame with both slots detected, the slots are a copy iff their
     2D boxes overlap (IoU >= COPY_IOU) and NOT both of: the left slot's wrist is nearer pose 15 than
     pose 16 by more than WRIST_MARGIN, and the right slot's wrist is nearer pose 16 than pose 15 by
     more than WRIST_MARGIN. Distances are in aspect-scaled image units divided by the clip's median
     shoulder width (pose 11 – 12). A copy is kept in one slot and the other is rejected (GHOST): with
     D_left = mean of both slot wrists' distances to pose 15 and D_right the same to pose 16, the
     left slot is kept iff D_left < D_right, else the right one; the frame is flagged
     `assignment_uncertain` unless min(D) / max(D) < ASSIGN_RATIO. Otherwise both are kept. Single
     hands are never moved between slots; a hand never detected stays absent.
  3. Coordinates relative to the pose nose (pose is present in every kept frame).
  4. Presence = the hand was detected and kept (KEPT). Stored, never re-inferred.
  5. Missing landmarks hold the nearest earlier detection of their part (the first one for leading
     frames); a part never detected stays 0.
  Status per frame and hand slot records provenance: ABSENT (not detected), KEPT, GHOST (detected,
  rejected by step 2); `assignment_uncertain` per frame marks a copy kept on weak wrist evidence.

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

CONTRACT_VERSION = "v4"  # part of the LMDB key (_cache_keys): old LMDBs fail loudly instead of mixing
ABSENT, KEPT, GHOST = 0, 1, 2
COPY_IOU, WRIST_MARGIN, ASSIGN_RATIO = 0.1, 0.1, 0.45  # step 2 (frozen: audit doc, round 2)
TARGET_FPS = 30  # training video and the browser's processing rate (step 0)


def frame_step_for(fps: float) -> int:
    """Raw frames per kept frame for a video recorded at fps: 120 → 4, 60 → 2, 24–44 → 1."""
    if not fps == fps or fps <= 0:
        raise ValueError(f"no frame rate ({fps})")
    return max(1, int(round(fps / TARGET_FPS)))

# Stored layout = config.ALL_COLUMNS: left hand, pose, right hand, selected face; x, y, z per landmark.
_PARTS = (("left_hand", N_LH), ("pose", N_POSE), ("right_hand", N_LH), ("face", len(SELECTED_FACE_INDICES)))
_FACE_POS = {lm: i for i, lm in enumerate(SELECTED_FACE_INDICES)}
_OWN_WRIST = (15, 16)  # pose landmark of the left / right wrist


_RAW_COUNT = {"pose": N_POSE, "left_hand": N_LH, "right_hand": N_LH, "face": 468}  # MediaPipe Holistic


def check_raw(df: pd.DataFrame) -> None:
    """Raises on raw rows that would still produce finite, well-shaped arrays but wrong ones: an
    unknown part type, a missing / non-integral frame id, a landmark index outside the part (a
    negative one would index from the end), or a repeated (frame, type, landmark_index) record."""
    bad_type = set(df["type"].unique()) - set(_RAW_COUNT)
    if bad_type:
        raise ValueError(f"unknown landmark types {sorted(map(str, bad_type))}")
    for col in ("frame", "landmark_index"):
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any() or (v != np.floor(v)).any():
            raise ValueError(f"{int((v.isna() | (v != np.floor(v))).sum())} rows with a missing or non-integral {col}")
    n = df["type"].map(_RAW_COUNT)
    out = (df["landmark_index"] < 0) | (df["landmark_index"] >= n)
    if out.any():
        r = df[out].iloc[0]
        raise ValueError(f"{int(out.sum())} landmark indices outside their part, e.g. {r['type']} {r['landmark_index']}")
    dup = df.duplicated(["frame", "type", "landmark_index"])
    if dup.any():
        r = df[dup].iloc[0]
        raise ValueError(f"{int(dup.sum())} repeated (frame, type, landmark_index) records, "
                         f"e.g. frame {r['frame']} {r['type']} {r['landmark_index']}")


def raw_arrays(df: pd.DataFrame) -> dict:
    """Competition-format rows (frame, type, landmark_index, x, y, z) → {part: (T, n, 3)} with NaN
    where a part was not detected, frames in order (`frame` holds their ids). Rows are checked
    first (check_raw)."""
    check_raw(df)
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


def resolve_ghosts(raw: dict, aspect: float = 0.75) -> tuple[np.ndarray, np.ndarray, dict]:
    """Step 2 on raw arrays (pose present in every frame) → status (T, 2) int8, assignment_uncertain
    (T,) bool and counts. aspect = frame width / height (distances are aspect-scaled)."""
    lh, rh, pose = raw["left_hand"], raw["right_hand"], raw["pose"]
    T = len(pose)
    det = np.stack([~np.isnan(lh[:, 0, 0]), ~np.isnan(rh[:, 0, 0])], 1)
    status = np.where(det, KEPT, ABSENT).astype(np.int8)
    uncertain = np.zeros(T, dtype=bool)
    both = np.nonzero(det.all(1))[0]
    copies = 0
    if len(both):
        scale = np.array([aspect, 1.0])
        sw = float(np.nanmedian(np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * scale, axis=-1)))
        sw = sw if sw > 0 else 1e-6

        def d(h, w, t):
            return float(np.linalg.norm((h[t, 0, :2] - pose[t, w, :2]) * scale)) / sw

        for t in both:
            if _iou(_box(lh[t]), _box(rh[t])) < COPY_IOU:
                continue
            l15, l16, r15, r16 = d(lh, 15, t), d(lh, 16, t), d(rh, 15, t), d(rh, 16, t)
            if l15 + WRIST_MARGIN < l16 and r16 + WRIST_MARGIN < r15:
                continue  # each slot on its own side: two hands
            d_left, d_right = (l15 + r15) / 2, (l16 + r16) / 2
            keep = 0 if d_left < d_right else 1
            status[t, 1 - keep] = GHOST
            hi = max(d_left, d_right)
            uncertain[t] = not (hi > 0 and min(d_left, d_right) / hi < ASSIGN_RATIO)
            copies += 1
    return status, uncertain, {"two_hand_frames": int(len(both)), "copy_frames": copies,
                               "assignment_uncertain_frames": int(uncertain.sum())}


def _hold_fill(a: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(T, ...) values; frames where present is False hold the nearest earlier present frame
    (the first present one before it); never present → 0."""
    if not present.any():
        return np.zeros_like(a)
    t = np.arange(len(a))
    src = np.maximum.accumulate(np.where(present, t, -1))
    src[src < 0] = int(np.argmax(present))
    return a[src]


def prepare(df: pd.DataFrame, aspect: float = 0.75, frame_step: int = 1) -> dict:
    """Steps 1–5: raw rows → {"coords": (T, COORD_FEAT) float32, "presence": (T, 2) uint8,
    "status": (T, 2) int8, "assignment_uncertain": (T,) uint8, "frames": kept frame ids, "stats": counts}.
    T can be 0 (no pose)."""
    raw = raw_arrays(df)
    inf = {part: int(np.isinf(raw[part]).sum()) for part, _ in _PARTS}  # every raw frame, before step 0
    if any(inf.values()):  # NaN means "not detected"; an infinity is corrupt data, never a value
        raise ValueError(f"infinite raw coordinates: { {k: v for k, v in inf.items() if v} }")
    if frame_step != 1:
        if type(frame_step) is not int or frame_step < 1:
            raise ValueError(f"frame_step {frame_step!r} is not a positive integer")
        raw = {k: v[::frame_step] for k, v in raw.items()}  # step 0: positions 0, k, 2k, …
    posed = ~np.isnan(raw["pose"]).all((1, 2))
    stats = {"frames_raw": int(len(posed)), "frames_no_pose": int((~posed).sum())}
    raw = {k: v[posed] for k, v in raw.items()}
    status, uncertain, ghost_stats = resolve_ghosts(raw, aspect)
    stats.update(ghost_stats)
    presence = status == KEPT
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
        blocks.append(np.where(np.isnan(a), 0.0, a).reshape(len(a), -1))  # never-seen parts → 0
    coords = np.concatenate(blocks, 1).astype(np.float32) if len(origin) else np.zeros((0, sum(n for _, n in _PARTS) * COORDS_PER_LM), np.float32)
    return {"coords": coords, "presence": presence.astype(np.uint8), "status": status,
            "assignment_uncertain": uncertain.astype(np.uint8), "frames": raw["frame"], "stats": stats}


RECORD_KEYS = ("coords", "presence", "status", "assignment_uncertain")  # every stored field, in digest order
RECORD_DTYPES = {"coords": torch.float32, "presence": torch.uint8, "status": torch.int8,
                 "assignment_uncertain": torch.uint8}


def dump_clip(clip: dict) -> bytes:
    """Stored LMDB value: RECORD_KEYS (torch.save of a dict of tensors)."""
    buf = io.BytesIO()
    torch.save({"coords": torch.from_numpy(np.ascontiguousarray(clip["coords"])),
                "presence": torch.from_numpy(clip["presence"]).to(torch.uint8),
                "status": torch.from_numpy(clip["status"]).to(torch.int8),
                "assignment_uncertain": torch.from_numpy(clip["assignment_uncertain"]).to(torch.uint8)}, buf)
    return buf.getvalue()


def load_record(value: bytes) -> dict:
    """Stored LMDB value → {RECORD_KEYS} tensors as stored."""
    d = torch.load(io.BytesIO(bytes(value)), weights_only=True)
    if not isinstance(d, dict) or set(d) != set(RECORD_KEYS):
        raise ValueError("legacy LMDB value (not the current record layout): rebuild the LMDB with the current builders")
    wrong = {k: str(d[k].dtype) for k in RECORD_KEYS if d[k].dtype != RECORD_DTYPES[k]}
    if wrong:  # same bytes under another dtype would load as different values
        raise ValueError(f"stored dtypes {wrong} ≠ {({k: str(RECORD_DTYPES[k]) for k in wrong})}")
    return d


def content_digest(env, paths) -> str:
    """sha256 over (path, stored length key, RECORD_KEYS with dtype and shape) of every clip, in
    path order — what the clips contain and what loaders read about them, independent of the LMDB
    file layout. env: an open lmdb environment."""
    import hashlib

    from ._cache_keys import lmdb_key, lmdb_length_key

    h = hashlib.sha256()
    with env.begin(buffers=True) as txn:
        for path in sorted(paths):
            d, n = load_record(txn.get(lmdb_key(path))), txn.get(lmdb_length_key(path))
            if n is None:
                raise ValueError(f"no length key for {path}")
            h.update(path.encode() + b"\0" + bytes(n) + b"\0")
            for k in RECORD_KEYS:
                h.update(f"{k}:{d[k].dtype}:{tuple(d[k].shape)}\0".encode())
                h.update(d[k].numpy().tobytes())
    return h.hexdigest()[:16]


def load_clip(value: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Stored LMDB value → (coords (T, COORD_FEAT) float32, presence (T, 2) float32)."""
    d = load_record(value)
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
