import numpy as np
import pandas as pd

def wrist_motion_features(positions, fps):
    """
    positions: numpy array of shape (T, 3) -> [ [x0,y0,z0], [x1,y1,z1], ... ]
    fps: frames per second (float)
    """
    positions = np.asarray(positions, dtype=float)
    T = positions.shape[0]
    dt = 1.0 / fps

    # 1. Displacement between consecutive frames
    disp = positions[1:] - positions[:-1]        # shape: (T-1, 3)
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]

    # 2. Distance & speed
    dist = np.linalg.norm(disp, axis=1)         # (T-1,)
    speed = dist / dt                           # (T-1,)

    # Avoid division by zero
    eps = 1e-8
    safe_dist = np.maximum(dist, eps)

    # 3. Unit direction
    direction = disp / safe_dist[:, None]       # (T-1, 3)

    # 4. Yaw & Pitch
    yaw = np.arctan2(dy, dx)                    # (T-1,)
    pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

    # 5. Axis angles (optional)
    alpha = np.arccos(dx / safe_dist)           # angle w.r.t X
    beta  = np.arccos(dy / safe_dist)           # angle w.r.t Y
    gamma = np.arccos(dz / safe_dist)           # angle w.r.t Z

    # 6. Turning angle between consecutive displacements
    v1 = disp[:-1]
    v2 = disp[1:]
    dot = np.sum(v1 * v2, axis=1)
    denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + eps
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    turning_angle = np.arccos(cos_theta)        # (T-2,)

    # 7. Yaw / Pitch change
    def wrap_angle(a):
        # wrap to (-pi, pi]
        return (a + np.pi) % (2 * np.pi) - np.pi

    delta_yaw = wrap_angle(yaw[1:] - yaw[:-1])      # (T-2,)
    delta_pitch = pitch[1:] - pitch[:-1]            # (T-2,)

    # 8. Velocity & acceleration
    velocity = disp / dt                         # (T-1, 3)
    vel_mag = np.linalg.norm(velocity, axis=1)   # (T-1,)

    acc = (velocity[1:] - velocity[:-1]) / dt    # (T-2, 3)
    acc_mag = np.linalg.norm(acc, axis=1)        # (T-2,)

    # 9. Jerk (optional)
    jerk = (acc[1:] - acc[:-1]) / dt             # (T-3, 3)
    jerk_mag = np.linalg.norm(jerk, axis=1)      # (T-3,)

    return {

        "distance": dist,            # (T-1,)
        "speed": speed,              # (T-1,)
        "yaw": yaw,                  # (T-1,)
        "pitch": pitch,              # (T-1,)
        "axis_angle_x": alpha,       # (T-1,)
        "axis_angle_y": beta,        # (T-1,)
        "axis_angle_z": gamma,       # (T-1,)
        "velocity_mag": vel_mag,          # (T-1,)
    }

landmark_data = pd.read_parquet("Kaggle_Data/train_landmark_files/16069/100015657.parquet")

landmark_consolidated = dict()

for landmark_type, landmark_type_data in landmark_data.groupby("type"):
    
    if landmark_type == "face":
        continue

    data_consolidation = []

    for landmark_id, landmark_id_data in landmark_type_data.groupby("landmark_index"):

        coordinates = landmark_id_data[['x', 'y', 'z']].values
        features_dict = motion_features(coordinates, 30)

        features_array = []
        for _, array in features_dict.items():
            features_array.append(array)

        features_array = np.array(features_array)
        data_consolidation.append(features_array)

    landmark_consolidated[landmark_type] = np.array(data_consolidation)


for k,v in landmark_consolidated.items():
    x,y,z = v.shape
    w = v.transpose(2,0,1).reshape(z, x*y)
    landmark_consolidated[k] = w
    

for k,v in landmark_consolidated.items():
    print(k)
    print(v.shape)

final_data = np.hstack(list(landmark_consolidated.values()))