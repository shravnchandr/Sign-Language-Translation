import json
import os
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    # config.py lives at research/models/cnn_transformer/ — go up 3 levels to project root
    _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    _LOCAL_BASE = os.path.join(_PROJECT_ROOT, "data", "Isolated_ASL_Recognition")
except NameError:
    _LOCAL_BASE = "/kaggle/input/competitions/asl-signs"

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

SELECTED_FACE_INDICES = []
for feature_indices in FACE_LANDMARK_INDICES.values():
    SELECTED_FACE_INDICES.extend(feature_indices)
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
# Eyebrows (grammatical: questions/negation) and mouth (phonological: mouthing)
# are split so the model can learn them with separate projections.
N_FACE_EYEBROW = len(FACE_LANDMARK_INDICES["left_eyebrow"]) + len(
    FACE_LANDMARK_INDICES["right_eyebrow"]
)
N_FACE_MOUTH = N_FACE - N_FACE_EYEBROW

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
