"""Robust preprocessing for MediaPipe landmarks with fallback normalization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# MediaPipe landmark indices
POSE_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FACE_LANDMARKS = 478  # Full MediaPipe face mesh


@dataclass
class LandmarkConfig:
    """Configuration for landmark processing."""

    include_z: bool = True
    face_subset: Optional[List[int]] = None  # None = use all 478
    interpolate_missing: bool = True
    max_missing_ratio: float = 0.3  # Max ratio of missing frames before rejection


class RobustPreprocessor:
    """
    Robust normalization with fallback chain: nose -> shoulder center -> hip center.

    Addresses blindspot #7: Fragile nose normalization.
    """

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.config = config or LandmarkConfig()

    def _get_pose_landmark(
        self, df: pd.DataFrame, frame: int, landmark_idx: int
    ) -> Optional[np.ndarray]:
        """Get a specific pose landmark for a frame."""
        mask = (
            (df["frame"] == frame)
            & (df["type"] == "pose")
            & (df["landmark_index"] == landmark_idx)
        )
        rows = df[mask]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        coords = [row["x"], row["y"]]
        if self.config.include_z:
            coords.append(row["z"])
        return np.array(coords)

    def _get_normalization_origin(
        self, df: pd.DataFrame, frame: int
    ) -> Tuple[np.ndarray, str]:
        """
        Get normalization origin with fallback chain.

        Returns:
            Tuple of (origin coordinates, origin name)
        """
        # Try nose first
        nose = self._get_pose_landmark(df, frame, POSE_INDICES["nose"])
        if nose is not None and not np.any(np.isnan(nose)):
            return nose, "nose"

        # Fallback to shoulder center
        left_shoulder = self._get_pose_landmark(
            df, frame, POSE_INDICES["left_shoulder"]
        )
        right_shoulder = self._get_pose_landmark(
            df, frame, POSE_INDICES["right_shoulder"]
        )

        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            if not np.any(np.isnan(shoulder_center)):
                return shoulder_center, "shoulder_center"

        # Fallback to hip center
        left_hip = self._get_pose_landmark(df, frame, POSE_INDICES["left_hip"])
        right_hip = self._get_pose_landmark(df, frame, POSE_INDICES["right_hip"])

        if left_hip is not None and right_hip is not None:
            hip_center = (left_hip + right_hip) / 2
            if not np.any(np.isnan(hip_center)):
                return hip_center, "hip_center"

        # Final fallback: return zeros (no normalization for this frame)
        n_coords = 3 if self.config.include_z else 2
        return np.zeros(n_coords), "none"

    def normalize_frame(self, df: pd.DataFrame, frame: int) -> pd.DataFrame:
        """Normalize a single frame's coordinates."""
        origin, _ = self._get_normalization_origin(df, frame)

        frame_df = df[df["frame"] == frame].copy()

        frame_df["x"] = frame_df["x"] - origin[0]
        frame_df["y"] = frame_df["y"] - origin[1]
        if self.config.include_z and len(origin) > 2:
            frame_df["z"] = frame_df["z"] - origin[2]

        return frame_df

    def normalize_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all frames in a sequence."""
        frames = df["frame"].unique()
        normalized_frames = []

        for frame in frames:
            normalized_frames.append(self.normalize_frame(df, frame))

        return pd.concat(normalized_frames, ignore_index=True)


class LandmarkProcessor:
    """
    Process raw parquet landmark data into model-ready tensors.

    Handles:
    - Robust normalization
    - Landmark separation (hands, pose, face)
    - Missing data interpolation
    - Z-coordinate inclusion (blindspot #1)
    """

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.config = config or LandmarkConfig()
        self.normalizer = RobustPreprocessor(config)

    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load and validate parquet file."""
        df = pd.read_parquet(path)
        required_cols = ["frame", "type", "landmark_index", "x", "y"]
        if self.config.include_z:
            required_cols.append("z")

        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def _extract_landmarks(
        self,
        df: pd.DataFrame,
        landmark_type: str,
        n_landmarks: int,
        subset_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Extract landmarks of a specific type into a structured array.

        Returns:
            Array of shape (T, N, C) where T=frames, N=landmarks, C=coordinates
        """
        type_df = df[df["type"] == landmark_type].copy()

        if subset_indices is not None:
            type_df = type_df[type_df["landmark_index"].isin(subset_indices)]
            n_landmarks = len(subset_indices)

        frames = sorted(df["frame"].unique())
        n_frames = len(frames)
        n_coords = 3 if self.config.include_z else 2

        # Initialize output array with NaN for missing detection
        output = np.full((n_frames, n_landmarks, n_coords), np.nan)

        frame_to_idx = {f: i for i, f in enumerate(frames)}

        for _, row in type_df.iterrows():
            frame_idx = frame_to_idx.get(row["frame"])
            if frame_idx is None:
                continue

            landmark_idx = row["landmark_index"]
            if subset_indices is not None:
                landmark_idx = (
                    subset_indices.index(landmark_idx)
                    if landmark_idx in subset_indices
                    else None
                )
                if landmark_idx is None:
                    continue

            if landmark_idx < n_landmarks:
                coords = [row["x"], row["y"]]
                if self.config.include_z:
                    coords.append(row["z"])
                output[frame_idx, landmark_idx] = coords

        return output

    def _interpolate_missing(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate missing values (NaN) along time axis."""
        if not self.config.interpolate_missing:
            return np.nan_to_num(arr, nan=0.0)

        T, N, C = arr.shape
        result = arr.copy()

        for n in range(N):
            for c in range(C):
                col = result[:, n, c]
                mask = ~np.isnan(col)

                if mask.sum() == 0:
                    # All missing, fill with zeros
                    result[:, n, c] = 0.0
                elif mask.sum() < T:
                    # Interpolate
                    x_valid = np.where(mask)[0]
                    y_valid = col[mask]
                    x_all = np.arange(T)
                    result[:, n, c] = np.interp(x_all, x_valid, y_valid)

        return result

    def process(self, path: str) -> Dict[str, np.ndarray]:
        """
        Process a parquet file into model-ready tensors.

        Args:
            path: Path to parquet file

        Returns:
            Dictionary with keys:
            - 'left_hand': (T, 21, 3)
            - 'right_hand': (T, 21, 3)
            - 'pose': (T, 33, 3)
            - 'face': (T, N_face, 3) where N_face depends on config
        """
        df = self.load_parquet(path)

        # Normalize first
        df = self.normalizer.normalize_sequence(df)

        # Extract each landmark type
        left_hand = self._extract_landmarks(df, "left_hand", HAND_LANDMARKS)
        right_hand = self._extract_landmarks(df, "right_hand", HAND_LANDMARKS)
        pose = self._extract_landmarks(df, "pose", POSE_LANDMARKS)

        # Face with optional subset
        if self.config.face_subset is not None:
            face = self._extract_landmarks(
                df,
                "face",
                len(self.config.face_subset),
                subset_indices=self.config.face_subset,
            )
        else:
            face = self._extract_landmarks(df, "face", FACE_LANDMARKS)

        # Interpolate missing values
        result = {
            "left_hand": self._interpolate_missing(left_hand),
            "right_hand": self._interpolate_missing(right_hand),
            "pose": self._interpolate_missing(pose),
            "face": self._interpolate_missing(face),
        }

        return result

    def process_to_flat(self, path: str) -> np.ndarray:
        """
        Process and flatten to a single tensor.

        Returns:
            Array of shape (T, N_total, 3) where N_total = 21+21+33+N_face
        """
        data = self.process(path)

        # Concatenate along landmark dimension
        return np.concatenate(
            [
                data["left_hand"],
                data["right_hand"],
                data["pose"],
                data["face"],
            ],
            axis=1,
        )


# Commonly used face landmark subsets
FACE_LANDMARK_SUBSETS = {
    "lips": list(range(0, 40)),  # Inner and outer lips
    "eyes": list(range(33, 133)) + list(range(263, 363)),  # Both eyes
    "eyebrows": [70, 63, 105, 66, 107, 55, 65, 52]
    + [300, 293, 334, 296, 336, 285, 295, 282],
    "nose": [1, 2, 4, 5, 6, 19, 94, 168, 197, 195],
    # Compact subset for efficient processing (134 landmarks)
    "compact": (
        [1, 2, 4, 5, 6, 19, 94, 168, 197, 195]  # nose (10)
        + [
            33,
            133,
            160,
            159,
            158,
            157,
            173,
            144,
            145,
            153,
            154,
            155,
            156,
            246,
            7,
            163,
        ]  # left eye (16)
        + [
            263,
            362,
            387,
            386,
            385,
            384,
            398,
            373,
            374,
            380,
            381,
            382,
            466,
            388,
            390,
            249,
        ]  # right eye (16)
        + [70, 63, 105, 66, 107, 55, 65, 52]  # left eyebrow (8)
        + [300, 293, 334, 296, 336, 285, 295, 282]  # right eyebrow (8)
        + [
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
        ]  # outer mouth (20)
        + [
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
        ]  # inner mouth (20)
        + [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ]  # face oval (36)
    ),
}
