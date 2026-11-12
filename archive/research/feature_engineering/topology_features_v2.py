import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# -----------------------------
# Configuration
# -----------------------------

# Pose landmark connections (MediaPipe Pose format)
# Adjust indices based on your specific pose model
POSE_CONNECTIONS = {
    # Upper body angles
    'left_elbow': ('pose', 11, 13, 15),      # shoulder-elbow-wrist
    'right_elbow': ('pose', 12, 14, 16),
    'left_shoulder': ('pose', 13, 11, 23),   # elbow-shoulder-hip
    'right_shoulder': ('pose', 14, 12, 24),
    
    # Lower body angles
    'left_knee': ('pose', 23, 25, 27),       # hip-knee-ankle
    'right_knee': ('pose', 24, 26, 28),
    'left_hip': ('pose', 11, 23, 25),        # shoulder-hip-knee
    'right_hip': ('pose', 12, 24, 26),
    
    # Core angles
    'spine': ('pose', 11, 23, 25),           # upper to lower body
}

# Important pairwise distances
POSE_DISTANCES = {
    'shoulder_width': ('pose', 11, 12),
    'hip_width': ('pose', 23, 24),
    'torso_length': ('pose', 11, 23),
}

# Hand landmark pairs for gesture recognition
HAND_DISTANCES = {
    'thumb_index': (4, 8),      # thumb tip to index tip
    'index_middle': (8, 12),    # index tip to middle tip
    'palm_width': (5, 17),      # thumb base to pinky base
}

# -----------------------------
# Core Feature Functions
# -----------------------------

def motion_features(positions, fps, smooth_sigma=1.0):
    """
    Compute essential motion features with optimized calculations.
    
    Returns features of length (T-1) for consistency.
    """
    positions = np.asarray(positions, dtype=float)
    
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3)")
    
    T = positions.shape[0]
    # Removed check for T < 3 to allow processing of short sequences
    
    # Smoothing
    if smooth_sigma > 0 and T >= 3:
        positions = np.column_stack([
            gaussian_filter1d(positions[:, i], sigma=smooth_sigma)
            for i in range(3)
        ])

    dt = 1.0 / float(fps)
    eps = 1e-8

    # === Velocity (first derivative) ===
    if T < 2:
        # Not enough frames for velocity
        empty_feat = np.zeros(0)
        return {
            "speed": empty_feat,
            "velocity_x": empty_feat,
            "velocity_y": empty_feat,
            "velocity_z": empty_feat,
            "yaw": empty_feat,
            "pitch": empty_feat,
            "acceleration": empty_feat,
            "jerk": empty_feat,
            "curvature": empty_feat,
        }

    vel = np.diff(positions, axis=0)  # (T-1, 3)
    speed = np.linalg.norm(vel, axis=1) / dt
    vel_normed = vel / dt
    
    # Direction angles
    yaw = np.arctan2(vel_normed[:, 1], vel_normed[:, 0])
    pitch = np.arctan2(vel_normed[:, 2], 
                       np.sqrt(vel_normed[:, 0]**2 + vel_normed[:, 1]**2))

    # === Acceleration (second derivative) ===
    if T >= 3:
        accel = np.diff(vel, axis=0) / (dt**2)  # (T-2, 3)
        accel_mag = np.linalg.norm(accel, axis=1)
        accel_mag = np.pad(accel_mag, (0, 1), mode='edge')  # pad to (T-1,)
    else:
        accel_mag = np.zeros(T-1)
        # Need accel for curvature calculation if we were to try, but T<3 implies no curvature either

    # === Jerk (third derivative) - smoothness indicator ===
    if T >= 4:
        jerk = np.diff(accel, axis=0) / dt  # (T-3, 3)
        jerk_mag = np.linalg.norm(jerk, axis=1)
        jerk_mag = np.pad(jerk_mag, (0, 2), mode='edge')  # pad to (T-1,)
    else:
        jerk_mag = np.zeros(T-1)

    # === Curvature - how much the path bends ===
    # Using discrete curvature: |v × a| / |v|^3
    if T >= 3:
        cross_prod = np.cross(vel[:-1], accel)  # (T-2, 3)
        curvature = np.linalg.norm(cross_prod, axis=1) / np.maximum(speed[:-1]**3, eps)
        curvature = np.pad(curvature, (0, 1), mode='edge')  # pad to (T-1,)
    else:
        curvature = np.zeros(T-1)

    return {
        # Velocity features (4)
        "speed": speed,
        "velocity_x": vel_normed[:, 0],
        "velocity_y": vel_normed[:, 1],
        "velocity_z": vel_normed[:, 2],
        
        # Direction features (2)
        "yaw": yaw,
        "pitch": pitch,
        
        # Higher-order motion (3)
        "acceleration": accel_mag,
        "jerk": jerk_mag,
        "curvature": curvature,
    }


def compute_angles_vectorized(landmarks, connections_dict):
    """
    Vectorized computation of joint angles.
    
    Parameters
    ----------
    landmarks : dict
        {landmark_type: (T, num_landmarks, 3)}
    connections_dict : dict
        {angle_name: (landmark_type, idx1, idx2, idx3)}
    
    Returns
    -------
    angles : dict
        {angle_name: (T,)}
    """
    angles = {}
    
    for angle_name, (ltype, idx1, idx2, idx3) in connections_dict.items():
        if ltype not in landmarks:
            continue
            
        coords = landmarks[ltype]
        if coords.shape[1] <= max(idx1, idx2, idx3):
            continue
        
        # Vectors from center joint to endpoints
        v1 = coords[:, idx1, :] - coords[:, idx2, :]  # (T, 3)
        v2 = coords[:, idx3, :] - coords[:, idx2, :]
        
        # Angle via dot product
        dot_prod = np.sum(v1 * v2, axis=1)
        mag_v1 = np.linalg.norm(v1, axis=1)
        mag_v2 = np.linalg.norm(v2, axis=1)
        
        cos_angle = np.clip(
            dot_prod / np.maximum(mag_v1 * mag_v2, 1e-8),
            -1.0, 1.0
        )
        angles[angle_name] = np.arccos(cos_angle)
    
    return angles


def compute_distances_vectorized(landmarks, distances_dict, landmark_type):
    """
    Vectorized computation of pairwise distances.
    
    Parameters
    ----------
    landmarks : dict
        {landmark_type: (T, num_landmarks, 3)}
    distances_dict : dict
        {distance_name: (idx1, idx2)} or {distance_name: (ltype, idx1, idx2)}
    landmark_type : str
        Type to compute distances for
    
    Returns
    -------
    distances : dict
        {distance_name: (T,)}
    """
    distances = {}
    
    if landmark_type not in landmarks:
        return distances
    
    coords = landmarks[landmark_type]
    
    for dist_name, indices in distances_dict.items():
        # Handle both (idx1, idx2) and (ltype, idx1, idx2) formats
        if len(indices) == 3:
            ltype, idx1, idx2 = indices
            if ltype != landmark_type:
                continue
        else:
            idx1, idx2 = indices
        
        if coords.shape[1] <= max(idx1, idx2):
            continue
        
        p1 = coords[:, idx1, :]
        p2 = coords[:, idx2, :]
        dist = np.linalg.norm(p1 - p2, axis=1)
        distances[dist_name] = dist
    
    return distances


def compute_relative_features(landmarks, reference_type='pose', reference_idx=0):
    """
    Compute position relative to reference point (e.g., nose or hip center).
    Returns normalized relative positions for scale invariance.
    
    Parameters
    ----------
    landmarks : dict
        {landmark_type: (T, num_landmarks, 3)}
    reference_type : str
        Landmark type containing reference point
    reference_idx : int
        Index of reference landmark
    
    Returns
    -------
    rel_features : dict
        {f"{ltype}_{idx}_rel_{axis}": (T,)}
    """
    rel_features = {}
    
    if reference_type not in landmarks:
        return rel_features
    
    ref_point = landmarks[reference_type][:, reference_idx, :]  # (T, 3)
    
    # Compute scale factor (e.g., torso length for normalization)
    if reference_type in landmarks:
        coords = landmarks[reference_type]
        if coords.shape[1] > 1:
            # Use distance between first two landmarks as scale
            scale = np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=1, keepdims=True)
            scale = np.maximum(scale, 1e-8)
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    for ltype, coords in landmarks.items():
        num_landmarks = coords.shape[1]
        
        # Only compute for a subset to avoid too many features
        # e.g., every 3rd landmark for hands, all for pose
        step = 1 if ltype == 'pose' else 3
        
        for idx in range(0, num_landmarks, step):
            rel_pos = (coords[:, idx, :] - ref_point) / scale  # (T, 3)
            
            # Only use magnitude and one dominant direction to reduce features
            rel_dist = np.linalg.norm(rel_pos, axis=1)
            rel_features[f"{ltype}_{idx}_rel_dist"] = rel_dist
            
            # Add dominant axis (the one with largest absolute mean)
            dominant_axis = np.argmax(np.abs(np.mean(rel_pos, axis=0)))
            axis_name = ['x', 'y', 'z'][dominant_axis]
            rel_features[f"{ltype}_{idx}_rel_{axis_name}"] = rel_pos[:, dominant_axis]
    
    return rel_features


def compute_symmetry_features(landmarks, landmark_type='pose'):
    """
    Compute left-right symmetry features (useful for detecting imbalanced poses).
    
    Parameters
    ----------
    landmarks : dict
    landmark_type : str
    
    Returns
    -------
    symmetry : dict
    """
    symmetry = {}
    
    if landmark_type not in landmarks:
        return symmetry
    
    coords = landmarks[landmark_type]
    
    # Pairs of left-right landmarks (MediaPipe Pose indices)
    lr_pairs = [
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
        (25, 26),  # knees
        (27, 28),  # ankles
    ]
    
    for left_idx, right_idx in lr_pairs:
        if coords.shape[1] <= max(left_idx, right_idx):
            continue
        
        left_pos = coords[:, left_idx, :]
        right_pos = coords[:, right_idx, :]
        
        # Y-coordinate difference (vertical asymmetry)
        y_diff = left_pos[:, 1] - right_pos[:, 1]
        symmetry[f"lr_y_diff_{left_idx}_{right_idx}"] = y_diff
    
    return symmetry


# -----------------------------
# Main Processing Pipeline
# -----------------------------

def extract_features(landmark_data, fps=30.0, smooth_sigma=1.0, 
                     include_hand_features=True):
    """
    Extract optimized feature set for action/gesture recognition.
    
    Parameters
    ----------
    landmark_data : pd.DataFrame
    fps : float
    smooth_sigma : float
    include_hand_features : bool
        Whether to include detailed hand features (set False for pose-only)
    
    Returns
    -------
    feature_matrix : np.ndarray of shape (T-1, num_features)
    feature_names : list of str
    """
    
    # Sort data
    sort_cols = [c for c in ["frame", "time"] if c in landmark_data.columns]
    if sort_cols:
        landmark_data = landmark_data.sort_values(sort_cols + ["type", "landmark_index"])
    
    # Organize data: type -> (T, num_landmarks, 3)
    landmarks_by_type = {}
    
    for landmark_type, type_data in landmark_data.groupby("type", sort=False):
        if landmark_type == "face":
            continue
        
        landmark_ids = sorted(type_data["landmark_index"].unique())
        coords_list = []
        
        for landmark_id in landmark_ids:
            coords = type_data[type_data["landmark_index"] == landmark_id][["x", "y", "z"]].to_numpy()
            coords_list.append(coords)
        
    # Stack: (num_landmarks, T, 3) -> (T, num_landmarks, 3)
        landmarks_by_type[landmark_type] = np.stack(coords_list, axis=0).transpose(1, 0, 2)
        
        # === Interpolate missing data (short gaps) ===
        # We use pandas interpolate on the stacked array
        # Reshape to (T, num_landmarks * 3) for easier dataframe interpolation
        T, num_landmarks, _ = landmarks_by_type[landmark_type].shape
        flat_data = landmarks_by_type[landmark_type].reshape(T, -1)
        
        # Interpolate with a limit to only fix short glitches (e.g., 2 frames)
        # 'linear' is good for motion
        flat_df = pd.DataFrame(flat_data)
        flat_df = flat_df.interpolate(method='linear', limit=2, limit_direction='both')
        
        # Reshape back
        landmarks_by_type[landmark_type] = flat_df.to_numpy().reshape(T, num_landmarks, 3)
    
    all_features = {}
    
    # === 1. Per-joint motion features ===
    for ltype, coords in landmarks_by_type.items():
        T, num_landmarks, _ = coords.shape
        
        # Compute for every joint (or subsample for hands if too many)
        step = 1 if ltype == 'pose' else 4  # every 4th hand landmark
        
        for idx in range(0, num_landmarks, step):
            joint_coords = coords[:, idx, :]
            feats = motion_features(joint_coords, fps, smooth_sigma)
            
            for feat_name, feat_values in feats.items():
                all_features[f"{ltype}_{idx}_{feat_name}"] = feat_values
    
    # === 2. Joint angles ===
    angles = compute_angles_vectorized(landmarks_by_type, POSE_CONNECTIONS)
    all_features.update(angles)
    
    # === 3. Important distances ===
    pose_dists = compute_distances_vectorized(landmarks_by_type, POSE_DISTANCES, 'pose')
    all_features.update(pose_dists)
    
    if include_hand_features:
        for hand_type in ['left_hand', 'right_hand']:
            if hand_type in landmarks_by_type:
                hand_dists = compute_distances_vectorized(
                    landmarks_by_type, HAND_DISTANCES, hand_type
                )
                # Prefix with hand type
                hand_dists = {f"{hand_type}_{k}": v for k, v in hand_dists.items()}
                all_features.update(hand_dists)
    
    # === 4. Relative position features (normalized) ===
    rel_feats = compute_relative_features(landmarks_by_type)
    all_features.update(rel_feats)
    
    # === 5. Symmetry features ===
    symmetry = compute_symmetry_features(landmarks_by_type, 'pose')
    all_features.update(symmetry)
    
    # === 6. Temporal derivatives of key angles ===
    # Angular velocity of important joints
    for angle_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
        if angle_name in all_features:
            angle_vals = all_features[angle_name]
            angular_vel = np.diff(angle_vals) * fps
            angular_vel = np.pad(angular_vel, (0, 1), mode='edge')
            all_features[f"{angle_name}_angular_vel"] = angular_vel
    
    # Stack all features
    feature_names = sorted(all_features.keys())
    feature_arrays = [all_features[name] for name in feature_names]
    
    # Ensure all same length
    min_len = min(len(f) for f in feature_arrays)
    feature_arrays = [f[:min_len] for f in feature_arrays]
    
    feature_matrix = np.column_stack(feature_arrays)
    
    # === Final Cleanup: Replace NaNs with 0 ===
    # This handles the "long gaps" where interpolation didn't fill values.
    # We treat missing data as "neutral/zero" features rather than noise.
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_matrix, feature_names


# -----------------------------
# Usage Example
# -----------------------------

if __name__ == "__main__":
    FPS = 30.0
    SMOOTH_SIGMA = 1.0
    
    # Load data
    landmark_data = pd.read_parquet(
        "Kaggle_Data/train_landmark_files/16069/100015657.parquet"
    )
    
    # Extract features
    feature_matrix, feature_names = extract_features(
        landmark_data, 
        fps=FPS, 
        smooth_sigma=SMOOTH_SIGMA,
        include_hand_features=True
    )
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Frames: {feature_matrix.shape[0]}")
    
    # Show feature breakdown
    print("\nFeature breakdown:")
    motion_feats = sum(1 for n in feature_names if any(x in n for x in ['speed', 'velocity', 'yaw', 'pitch', 'acceleration', 'jerk', 'curvature']))
    angle_feats = sum(1 for n in feature_names if 'elbow' in n or 'knee' in n or 'hip' in n or 'shoulder' in n)
    dist_feats = sum(1 for n in feature_names if 'width' in n or 'length' in n or 'thumb' in n)
    rel_feats = sum(1 for n in feature_names if 'rel_' in n)
    sym_feats = sum(1 for n in feature_names if 'lr_' in n)
    
    print(f"  Motion features: {motion_feats}")
    print(f"  Angle features: {angle_feats}")
    print(f"  Distance features: {dist_feats}")
    print(f"  Relative position features: {rel_feats}")
    print(f"  Symmetry features: {sym_feats}")
    
    # Show first 20 feature names
    print(f"\nFirst 20 features:")
    for i, name in enumerate(feature_names[:20]):
        print(f"  {i+1:2d}. {name}")
    
    # Basic statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {np.mean(feature_matrix, axis=0)[:5]}")
    print(f"  Std:  {np.std(feature_matrix, axis=0)[:5]}")
    print(f"  Min:  {np.min(feature_matrix, axis=0)[:5]}")
    print(f"  Max:  {np.max(feature_matrix, axis=0)[:5]}")