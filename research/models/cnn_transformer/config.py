import json
import os
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    # config.py lives at research/models/cnn_transformer/ — go up 3 levels to project root
    _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    _LOCAL_BASE = os.path.join(_PROJECT_ROOT, "data", "asl-is-lmdb")
except NameError:
    _LOCAL_BASE = "/kaggle/input/asl-is-lmdb"

BASE_PATH = os.environ.get("KAGGLE_INPUT_DIR", _LOCAL_BASE)
TRAIN_FILE = os.path.join(BASE_PATH, "train.csv")
SIGN_INDEX_FILE = os.path.join(BASE_PATH, "sign_to_prediction_index_map.json")

if os.path.exists(SIGN_INDEX_FILE):
    with open(SIGN_INDEX_FILE, "r") as json_file:
        SIGN2INDEX_JSON = json.load(json_file)
else:
    SIGN2INDEX_JSON = {}

INCLUDE_FACE = True
INCLUDE_DEPTH = True

FACE_LANDMARK_INDICES = {
    # Eyebrows carry facial grammar signal in ASL (raised = question, furrowed = negation)
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282],
    # Lips are essential for mouthing components and mouth-shape signs
    "mouth_outer": [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
    ],
    "mouth_inner": [
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ],
    # Removed: nose (10), left_eye (16), right_eye (16), face_oval (36).
    # Nose/eyes/oval encode head geometry (signer identity), not sign content.
}

# Eyebrows MUST come before mouth so the FACE_START : FACE_START+N_FACE_EYEBROW
# slice in landmark_conformer.py maps to the correct anatomical group.
# Explicitly ordered here so a future dict reordering can't silently corrupt it.
_EYEBROW_KEYS = {"left_eyebrow", "right_eyebrow"}
SELECTED_FACE_INDICES = []
for _key in ("left_eyebrow", "right_eyebrow", "mouth_outer", "mouth_inner"):
    SELECTED_FACE_INDICES.extend(FACE_LANDMARK_INDICES[_key])
FACE_LANDMARK_SET = frozenset(SELECTED_FACE_INDICES)  # O(1) membership test


def generate_full_column_list() -> List[str]:
    landmark_specs = {"left_hand": 21, "pose": 33, "right_hand": 21}
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]
    full_columns = []
    for landmark_type, count in landmark_specs.items():
        for i in range(count):
            for axis in axes:
                full_columns.append(f"{landmark_type}_{i}_{axis}")
    if INCLUDE_FACE:
        for face_idx in SELECTED_FACE_INDICES:
            for axis in axes:
                full_columns.append(f"face_{face_idx}_{axis}")
    return full_columns


ALL_COLUMNS = generate_full_column_list()
COORDS_PER_LM = 3 if INCLUDE_DEPTH else 2
COORD_FEAT = len(ALL_COLUMNS)
LH_START, POSE_START, RH_START, FACE_START = (
    0,
    21 * COORDS_PER_LM,
    (21 + 33) * COORDS_PER_LM,
    (21 + 33 + 21) * COORDS_PER_LM,
)
N_LH, N_POSE, N_RH = 21, 33, 21
N_FACE = len(SELECTED_FACE_INDICES)

# Model input layout: [pos (COORD_FEAT) | Δ1 vel (COORD_FEAT) | presence (2)].
# Presence = per-frame (lh, rh) detection flags derived in the datasets by
# hand_presence(); they let the model tell a missing hand from one at the nose.
N_PRESENCE = 2
PRESENCE_START = 2 * COORD_FEAT
IN_FEAT = PRESENCE_START + N_PRESENCE
# Eyebrows (grammatical: questions/negation) and mouth (phonological: mouthing)
# are split so the model can learn them with separate projections.
N_FACE_EYEBROW = len(FACE_LANDMARK_INDICES["left_eyebrow"]) + len(
    FACE_LANDMARK_INDICES["right_eyebrow"]
)
N_FACE_MOUTH = N_FACE - N_FACE_EYEBROW

# ---------------------------------------------------------------------------
# Horizontal mirror: left↔right landmark pairs. A true mirror must negate x AND
# swap every bilateral landmark (hands, pose limbs, eyebrows, lip corners) —
# swapping only the hand blocks leaves pose/face anatomically inconsistent.
# Pairs verified on real frames: mirrored-pair error is 5–10× below a shuffled
# pairing for pose, eyebrows and mouth.
# ---------------------------------------------------------------------------
_POSE_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
_FACE_MIRROR_PAIRS = list(
    zip(FACE_LANDMARK_INDICES["left_eyebrow"], FACE_LANDMARK_INDICES["right_eyebrow"])
) + [
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
    (37, 267), (39, 269), (40, 270), (185, 409),  # outer lip
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    (87, 317), (178, 402), (88, 318), (95, 324),  # inner lip
]  # midline points (0, 17, 13, 14) map to themselves


def _build_mirror_perm() -> List[int]:
    """Feature-index permutation over [pos | vel | presence] (length IN_FEAT)."""
    n_lm = COORD_FEAT // COORDS_PER_LM
    lm_perm = list(range(n_lm))
    lh0, pose0, rh0 = 0, N_LH, N_LH + N_POSE
    for i in range(N_LH):
        lm_perm[lh0 + i], lm_perm[rh0 + i] = rh0 + i, lh0 + i
    for a, b in _POSE_MIRROR_PAIRS:
        lm_perm[pose0 + a], lm_perm[pose0 + b] = pose0 + b, pose0 + a
    face0 = N_LH + N_POSE + N_RH
    pos_of = {idx: face0 + k for k, idx in enumerate(SELECTED_FACE_INDICES)}
    for a, b in _FACE_MIRROR_PAIRS:
        lm_perm[pos_of[a]], lm_perm[pos_of[b]] = pos_of[b], pos_of[a]
    perm = []
    for half in (0, COORD_FEAT):
        for lm in lm_perm:
            perm.extend(half + lm * COORDS_PER_LM + k for k in range(COORDS_PER_LM))
    perm.extend([PRESENCE_START + 1, PRESENCE_START])  # lh_present ↔ rh_present
    return perm


MIRROR_PERM = _build_mirror_perm()
assert sorted(MIRROR_PERM) == list(range(IN_FEAT)), "mirror perm not a bijection"

FINGER_LM_RANGES: List[Tuple[int, int]] = [
    (1, 5),  # thumb
    (5, 9),  # index
    (9, 13),  # middle
    (13, 17),  # ring
    (17, 21),  # pinky
]


def get_finger_coord_slices():
    slices_dict = {}
    for hand_label, hand_start in (("left", LH_START), ("right", RH_START)):
        for fi, (lm_lo, lm_hi) in enumerate(FINGER_LM_RANGES):
            slices = []
            for half in (0, COORD_FEAT):
                feat_lo = half + hand_start + lm_lo * COORDS_PER_LM
                feat_hi = half + hand_start + lm_hi * COORDS_PER_LM
                slices.append((feat_lo, feat_hi))
            slices_dict[(hand_label, fi)] = slices
    return slices_dict


FINGER_COORD_SLICES = get_finger_coord_slices()
