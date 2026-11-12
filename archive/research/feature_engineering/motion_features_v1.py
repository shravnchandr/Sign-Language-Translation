import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

# -----------------------------
# 1. Enhanced motion feature function
# -----------------------------

def motion_features(positions, fps, smooth_sigma=1.0):
    """
    Compute comprehensive motion features for 3D joints across time.

    Parameters
    ----------
    positions : array-like of shape (T, 3)
        Joint coordinates over time.
    fps : float
        Frames per second.
    smooth_sigma : float, optional
        Gaussian smoothing sigma. Set to 0 to disable. Default: 1.0

    Returns
    -------
    features : dict of 1D np.ndarray, length (T-1) or (T-2) for acceleration
        Velocity-based: speed, velocity_vector_x/y/z, yaw, pitch
        Direction: axis_angle_x/y/z
        Acceleration: acceleration_mag, accel_yaw, accel_pitch
        Smoothness: jerk_mag
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3)")
    
    T = positions.shape[0]
    if T < 3:
        raise ValueError("Need at least 3 frames to compute acceleration features.")

    # Optional smoothing to reduce noise
    if smooth_sigma > 0:
        positions = np.column_stack([
            gaussian_filter1d(positions[:, i], sigma=smooth_sigma)
            for i in range(3)
        ])

    dt = 1.0 / float(fps)

    # 1. Velocity (displacement between consecutive frames)
    vel = positions[1:] - positions[:-1]  # (T-1, 3)
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    
    speed = np.linalg.norm(vel, axis=1)  # (T-1,)
    speed = speed / dt
    vel_x = vx / dt
    vel_y = vy / dt
    vel_z = vz / dt

    eps = 1e-8
    safe_speed = np.maximum(speed, eps)

    # 2. Direction angles (yaw, pitch)
    yaw = np.arctan2(vel_y, vel_x)
    pitch = np.arctan2(vel_z, np.sqrt(vel_x**2 + vel_y**2))

    # 3. Axis angles (direction w.r.t coordinate axes)
    cos_x = np.clip(vel_x / safe_speed, -1.0, 1.0)
    cos_y = np.clip(vel_y / safe_speed, -1.0, 1.0)
    cos_z = np.clip(vel_z / safe_speed, -1.0, 1.0)

    axis_angle_x = np.arccos(cos_x)
    axis_angle_y = np.arccos(cos_y)
    axis_angle_z = np.arccos(cos_z)

    # 4. Acceleration (change in velocity)
    accel = vel[1:] - vel[:-1]  # (T-2, 3)
    ax, ay, az = accel[:, 0], accel[:, 1], accel[:, 2]
    
    accel_mag = np.linalg.norm(accel, axis=1) / (dt**2)  # (T-2,)
    safe_accel = np.maximum(accel_mag, eps)

    # Acceleration direction angles
    accel_yaw = np.arctan2(ay / dt, ax / dt)
    accel_pitch = np.arctan2(az / dt, np.sqrt((ax/dt)**2 + (ay/dt)**2))

    # 5. Jerk (smoothness: rate of change of acceleration)
    jerk = accel[1:] - accel[:-1]  # (T-3, 3)
    jerk_mag = np.linalg.norm(jerk, axis=1) / (dt**3)  # (T-3,)

    # Pad features to match original sequence length for consistency
    # Using forward fill for shorter sequences
    accel_mag_padded = np.pad(accel_mag, (0, 1), mode='edge')  # (T-1,)
    jerk_mag_padded = np.pad(jerk_mag, (0, 2), mode='edge')    # (T-1,)
    accel_yaw_padded = np.pad(accel_yaw, (0, 1), mode='edge')
    accel_pitch_padded = np.pad(accel_pitch, (0, 1), mode='edge')

    return {
        # Velocity features
        "speed":         speed,
        "velocity_x":    vel_x,
        "velocity_y":    vel_y,
        "velocity_z":    vel_z,
        
        # Direction features
        "yaw":           yaw,
        "pitch":         pitch,
        "axis_angle_x":  axis_angle_x,
        "axis_angle_y":  axis_angle_y,
        "axis_angle_z":  axis_angle_z,
        
        # Acceleration features
        "acceleration":  accel_mag_padded,
        "accel_yaw":     accel_yaw_padded,
        "accel_pitch":   accel_pitch_padded,
        
        # Jerk (smoothness)
        "jerk":          jerk_mag_padded,
    }


def compute_joint_angles(landmarks_dict, joint_connections):
    """
    Compute angles between connected joints (e.g., elbow angle = angle(shoulder-elbow-wrist)).

    Parameters
    ----------
    landmarks_dict : dict
        Maps landmark_type -> (T, num_landmarks, 3) array of coordinates.
    joint_connections : list of tuples
        Each tuple is (landmark_type, idx1, idx2) or (landmark_type, idx1, idx2, idx3)
        For 3-point angle: angle at idx2 formed by idx1-idx2-idx3

    Returns
    -------
    joint_angles : dict
        Maps joint_name -> (T, 1) array of angles in radians
    """
    joint_angles = {}
    
    for conn in joint_connections:
        if len(conn) == 4:
            ltype, idx1, idx2, idx3 = conn
            p1 = landmarks_dict[ltype][:, idx1, :]  # (T, 3)
            p2 = landmarks_dict[ltype][:, idx2, :]
            p3 = landmarks_dict[ltype][:, idx3, :]
            
            # Vectors from joint to endpoints
            v1 = p1 - p2  # (T, 3)
            v2 = p3 - p2
            
            # Compute angle via dot product
            dot_prod = np.sum(v1 * v2, axis=1)  # (T,)
            mag_v1 = np.linalg.norm(v1, axis=1)
            mag_v2 = np.linalg.norm(v2, axis=1)
            
            cos_angle = np.clip(
                dot_prod / (np.maximum(mag_v1 * mag_v2, 1e-8)),
                -1.0, 1.0
            )
            angle = np.arccos(cos_angle)  # (T,)
            
            joint_name = f"{ltype}_{idx1}_{idx2}_{idx3}"
            joint_angles[joint_name] = angle

    return joint_angles


def compute_relative_positions(landmarks_dict, reference_joint=None):
    """
    Compute relative positions of joints w.r.t. a reference point (e.g., torso center).

    Parameters
    ----------
    landmarks_dict : dict
        Maps landmark_type -> (T, num_landmarks, 3) array.
    reference_joint : tuple, optional
        (landmark_type, landmark_idx). If None, use center of all landmarks.

    Returns
    -------
    relative_pos : dict
        Maps (landmark_type, idx) -> (T, 3) relative position array
    """
    relative_pos = {}
    
    if reference_joint is None:
        # Compute center across all landmarks
        all_landmarks = np.concatenate(list(landmarks_dict.values()), axis=1)
        center = np.mean(all_landmarks, axis=1, keepdims=True)  # (T, 1, 3)
    else:
        ref_type, ref_idx = reference_joint
        center = landmarks_dict[ref_type][:, ref_idx:ref_idx+1, :]  # (T, 1, 3)

    for ltype, coords in landmarks_dict.items():
        for idx in range(coords.shape[1]):
            rel = coords[:, idx, :] - center[:, 0, :]  # (T, 3)
            relative_pos[(ltype, idx)] = rel

    return relative_pos


# FPS and smoothing parameters
FPS = 30.0
SMOOTH_SIGMA = 1.0  # Gaussian smoothing; 0 to disable

# Load landmark data
landmark_data = pd.read_parquet(
    "Kaggle_Data/train_landmark_files/16069/100015657.parquet"
)

sort_cols = [c for c in ["frame", "time"] if c in landmark_data.columns]
if sort_cols:
    landmark_data = landmark_data.sort_values(sort_cols + ["type", "landmark_index"])

# =======================
# 2. Reorganize data structure for easier processing
# =======================

# landmarks_by_type: type -> (T, num_landmarks, 3)
landmarks_by_type = {}

for landmark_type, type_data in landmark_data.groupby("type", sort=False):
    if landmark_type == "face":
        continue

    landmark_ids = sorted(type_data["landmark_index"].unique())
    T = len(type_data) // len(landmark_ids)
    
    coords_list = []
    for landmark_id in landmark_ids:
        coords = type_data[type_data["landmark_index"] == landmark_id][["x", "y", "z"]].to_numpy()
        coords_list.append(coords)
    
    # Stack: (num_landmarks, T, 3) -> transpose to (T, num_landmarks, 3)
    landmarks_by_type[landmark_type] = np.stack(coords_list, axis=0).transpose(1, 0, 2)

# =======================
# 3. Compute motion features for each joint
# =======================

landmark_consolidated = {}
feature_keys = [
    "speed",
    "velocity_x", "velocity_y", "velocity_z",
    "yaw", "pitch",
    "axis_angle_x", "axis_angle_y", "axis_angle_z",
    "acceleration", "accel_yaw", "accel_pitch",
    "jerk",
]

for landmark_type, coords in landmarks_by_type.items():
    # coords: (T, num_landmarks, 3)
    T, num_landmarks, _ = coords.shape
    
    # Compute features for each landmark
    all_features = []
    
    for landmark_idx in range(num_landmarks):
        joint_coords = coords[:, landmark_idx, :]  # (T, 3)
        feats = motion_features(joint_coords, FPS, smooth_sigma=SMOOTH_SIGMA)
        
        # Stack features: (num_features, T-1)
        feat_matrix = np.stack([feats[k] for k in feature_keys], axis=0)
        all_features.append(feat_matrix)
    
    # (num_landmarks, num_features, T-1) -> (T-1, num_landmarks * num_features)
    all_features = np.stack(all_features, axis=0)  # (num_landmarks, num_features, T-1)
    T_minus_1 = all_features.shape[2]
    all_features = all_features.transpose(2, 0, 1).reshape(T_minus_1, num_landmarks * len(feature_keys))
    
    landmark_consolidated[landmark_type] = all_features
    print(f"{landmark_type}: landmarks={num_landmarks}, features={len(feature_keys)}, frames={T_minus_1}")
    print(f"  shape: {all_features.shape}")

final_data = np.hstack(list(landmark_consolidated.values()))
print(final_data.shape)