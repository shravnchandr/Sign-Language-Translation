"""
Graph Structure Definition for MediaPipe Landmarks

This module defines the skeletal connections (edges) between landmarks for:
- Left hand (21 landmarks)
- Right hand (21 landmarks)
- Pose (33 landmarks)
- Face (134 selected landmarks)

These edges form the spatial graph structure for ST-GCN.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict


# Face landmark indices (from data_prep.py) - defined at module level
FACE_LANDMARK_INDICES = {
    "nose": [1, 2, 4, 5, 6, 19, 94, 168, 197, 195],
    "left_eye": [
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
    ],
    "right_eye": [
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
    ],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282],
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
    "face_oval": [
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
    ],
}

# Flatten all selected indices
SELECTED_FACE_INDICES = []
for feature_indices in FACE_LANDMARK_INDICES.values():
    SELECTED_FACE_INDICES.extend(feature_indices)

# Create mapping from global face indices to local indices (0-133)
FACE_GLOBAL_TO_LOCAL = {
    global_idx: local_idx for local_idx, global_idx in enumerate(SELECTED_FACE_INDICES)
}


class LandmarkGraph:
    """
    Defines the graph structure for MediaPipe landmarks.

    The graph has two types of edges:
    1. Spatial edges: Connect anatomically related landmarks (e.g., finger joints)
    2. Temporal edges: Connect the same landmark across consecutive frames
    """

    def __init__(self, include_face=True):
        self.include_face = include_face

        # Total number of landmarks
        self.num_landmarks = (
            209 if include_face else 75
        )  # face(134) + hands(42) + pose(33)

        # Create face index mapping if using face landmarks
        if include_face:
            self.face_global_to_local = FACE_GLOBAL_TO_LOCAL
        else:
            self.face_global_to_local = {}

        # Define spatial edges for each body part
        self.hand_edges = self._get_hand_edges()
        self.pose_edges = self._get_pose_edges()
        self.face_edges = self._get_face_edges() if include_face else []
        self.cross_body_edges = self._get_cross_body_edges()

        # Combine all spatial edges
        self.spatial_edges = self._build_complete_spatial_graph()

        print(f"Graph structure created:")
        print(f"  - Total landmarks: {self.num_landmarks}")
        print(f"  - Spatial edges: {len(self.spatial_edges)}")
        print(f"  - Hand edges: {len(self.hand_edges) * 2}")  # left + right
        print(f"  - Pose edges: {len(self.pose_edges)}")
        print(f"  - Face edges: {len(self.face_edges)}")
        print(f"  - Cross-body edges: {len(self.cross_body_edges)}")

    def _get_hand_edges(self) -> List[Tuple[int, int]]:
        """
        Define the skeletal structure of a hand.

        Hand landmarks (0-20):
        - 0: Wrist
        - 1-4: Thumb (from base to tip)
        - 5-8: Index finger
        - 9-12: Middle finger
        - 13-16: Ring finger
        - 17-20: Pinky finger

        Returns:
            List of (source, target) tuples representing connections
        """
        edges = []

        # Thumb: wrist → base → joints → tip
        edges.extend([(0, 1), (1, 2), (2, 3), (3, 4)])

        # Index finger: wrist → base → joints → tip
        edges.extend([(0, 5), (5, 6), (6, 7), (7, 8)])

        # Middle finger
        edges.extend([(0, 9), (9, 10), (10, 11), (11, 12)])

        # Ring finger
        edges.extend([(0, 13), (13, 14), (14, 15), (15, 16)])

        # Pinky finger
        edges.extend([(0, 17), (17, 18), (18, 19), (19, 20)])

        # Palm connections (connect finger bases to form palm structure)
        # Creates a "web" between fingers
        edges.extend(
            [
                (5, 9),  # Index base to middle base
                (9, 13),  # Middle base to ring base
                (13, 17),  # Ring base to pinky base
            ]
        )

        return edges

    def _get_pose_edges(self) -> List[Tuple[int, int]]:
        """
        Define the skeletal structure of the pose/body.

        Key pose landmarks:
        - 0: Nose
        - 11, 12: Shoulders (left, right)
        - 13, 14: Elbows (left, right)
        - 15, 16: Wrists (left, right)
        - 23, 24: Hips (left, right)

        Returns:
            List of (source, target) tuples
        """
        edges = []

        # Face/Head structure
        edges.extend(
            [
                (0, 1),
                (1, 2),
                (2, 3),  # Nose to left eye
                (0, 4),
                (4, 5),
                (5, 6),  # Nose to right eye
                (0, 7),
                (0, 8),  # Nose to ears
            ]
        )

        # Mouth
        edges.append((9, 10))

        # Shoulder line
        edges.append((11, 12))

        # Left arm: shoulder → elbow → wrist
        edges.extend([(11, 13), (13, 15)])

        # Right arm: shoulder → elbow → wrist
        edges.extend([(12, 14), (14, 16)])

        # Torso: shoulders to hips
        edges.extend([(11, 23), (12, 24)])

        # Hip line
        edges.append((23, 24))

        # Legs (less important for ASL, but included for completeness)
        # Left leg: hip → knee → ankle → foot
        edges.extend([(23, 25), (25, 27), (27, 29), (29, 31)])

        # Right leg
        edges.extend([(24, 26), (26, 28), (28, 30), (30, 32)])

        return edges

    def _get_face_edges(self) -> List[Tuple[int, int]]:
        """
        Define connections for the 134 selected face landmarks.

        Returns edges using LOCAL indices (0-133), not global MediaPipe indices.
        """
        edges = []

        # Helper function to convert global indices to local
        def to_local(global_indices):
            return [self.face_global_to_local[idx] for idx in global_indices]

        # Nose connections
        nose_indices = to_local([1, 2, 4, 5, 6, 19, 94, 168, 197, 195])
        for i in range(len(nose_indices) - 1):
            edges.append((nose_indices[i], nose_indices[i + 1]))

        # Left eye contour (closed loop)
        left_eye = to_local(
            [
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
            ]
        )
        for i in range(len(left_eye)):
            edges.append((left_eye[i], left_eye[(i + 1) % len(left_eye)]))

        # Right eye contour (closed loop)
        right_eye = to_local(
            [
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
            ]
        )
        for i in range(len(right_eye)):
            edges.append((right_eye[i], right_eye[(i + 1) % len(right_eye)]))

        # Left eyebrow
        left_eyebrow = to_local([70, 63, 105, 66, 107, 55, 65, 52])
        for i in range(len(left_eyebrow) - 1):
            edges.append((left_eyebrow[i], left_eyebrow[i + 1]))

        # Right eyebrow
        right_eyebrow = to_local([300, 293, 334, 296, 336, 285, 295, 282])
        for i in range(len(right_eyebrow) - 1):
            edges.append((right_eyebrow[i], right_eyebrow[i + 1]))

        # Mouth outer contour (closed loop)
        mouth_outer = to_local(
            [
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
            ]
        )
        for i in range(len(mouth_outer)):
            edges.append((mouth_outer[i], mouth_outer[(i + 1) % len(mouth_outer)]))

        # Mouth inner contour (closed loop)
        mouth_inner = to_local(
            [
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
            ]
        )
        for i in range(len(mouth_inner)):
            edges.append((mouth_inner[i], mouth_inner[(i + 1) % len(mouth_inner)]))

        # Face oval (closed contour)
        face_oval = to_local(
            [
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
            ]
        )
        for i in range(len(face_oval)):
            edges.append((face_oval[i], face_oval[(i + 1) % len(face_oval)]))

        # Connect features to each other for information flow
        # Convert these connections to local indices too
        nose_to_left_eye = (self.face_global_to_local[1], self.face_global_to_local[33])
        nose_to_right_eye = (
            self.face_global_to_local[1],
            self.face_global_to_local[263],
        )
        left_eye_to_eyebrow = (
            self.face_global_to_local[33],
            self.face_global_to_local[70],
        )
        right_eye_to_eyebrow = (
            self.face_global_to_local[263],
            self.face_global_to_local[300],
        )
        nose_to_mouth = (self.face_global_to_local[1], self.face_global_to_local[61])

        edges.extend(
            [
                nose_to_left_eye,
                nose_to_right_eye,
                left_eye_to_eyebrow,
                right_eye_to_eyebrow,
                nose_to_mouth,
            ]
        )

        return edges

    def _get_cross_body_edges(self) -> List[Tuple[int, int]]:
        """
        Connect different body parts together.

        This allows information to flow between:
        - Hands and arms (wrists)
        - Face and head pose

        Landmark ordering in our data:
        - Face: 0-133 (if included)
        - Left hand: 134-154 (or 0-20 without face)
        - Pose: 155-187 (or 21-53 without face)
        - Right hand: 188-208 (or 54-74 without face)

        Returns:
            List of (source, target) tuples connecting body parts
        """
        edges = []

        if self.include_face:
            # With face landmarks
            left_hand_offset = 134
            pose_offset = 155
            right_hand_offset = 188

            # Connect hand wrists (index 0 in hand) to pose wrists (indices 15, 16)
            edges.extend(
                [
                    (
                        left_hand_offset + 0,
                        pose_offset + 15,
                    ),  # Left hand wrist to pose left wrist
                    (
                        right_hand_offset + 0,
                        pose_offset + 16,
                    ),  # Right hand wrist to pose right wrist
                ]
            )

            # Connect face nose (index 1) to pose nose (index 0)
            # Need to convert face global index 1 to local index
            face_nose_local = self.face_global_to_local[1]
            edges.append((face_nose_local, pose_offset + 0))

        else:
            # Without face landmarks
            left_hand_offset = 0
            pose_offset = 21
            right_hand_offset = 54

            # Connect hand wrists to pose wrists
            edges.extend(
                [
                    (
                        left_hand_offset + 0,
                        pose_offset + 15,
                    ),  # Left hand wrist to pose left wrist
                    (
                        right_hand_offset + 0,
                        pose_offset + 16,
                    ),  # Right hand wrist to pose right wrist
                ]
            )

        return edges

    def _build_complete_spatial_graph(self) -> List[Tuple[int, int]]:
        """
        Combine all edge definitions into a single spatial graph.

        Maps local indices (0-20 for hand, 0-32 for pose, etc.) to
        global indices in the full landmark array.

        Returns:
            List of (source, target) tuples with global indices
        """
        all_edges = []

        if self.include_face:
            # Landmark ordering: face(0-133), left_hand(134-154), pose(155-187), right_hand(188-208)
            left_hand_offset = 134
            pose_offset = 155
            right_hand_offset = 188

            # Add face edges (already use global indices 0-467, but we filtered to 134)
            all_edges.extend(self.face_edges)

        else:
            # Landmark ordering: left_hand(0-20), pose(21-53), right_hand(54-74)
            left_hand_offset = 0
            pose_offset = 21
            right_hand_offset = 54

        # Add left hand edges (offset by left_hand_offset)
        for src, dst in self.hand_edges:
            all_edges.append((src + left_hand_offset, dst + left_hand_offset))

        # Add pose edges (offset by pose_offset)
        for src, dst in self.pose_edges:
            all_edges.append((src + pose_offset, dst + pose_offset))

        # Add right hand edges (offset by right_hand_offset)
        for src, dst in self.hand_edges:
            all_edges.append((src + right_hand_offset, dst + right_hand_offset))

        # Add cross-body connections
        all_edges.extend(self.cross_body_edges)

        # Validate all edges are within bounds
        for src, dst in all_edges:
            if src >= self.num_landmarks or dst >= self.num_landmarks:
                raise ValueError(
                    f"Edge ({src}, {dst}) out of bounds! "
                    f"Max landmark index: {self.num_landmarks - 1}"
                )
            if src < 0 or dst < 0:
                raise ValueError(f"Edge ({src}, {dst}) has negative index!")

        return all_edges

    def get_adjacency_matrix(self, self_loops=True) -> np.ndarray:
        """
        Convert edge list to adjacency matrix.

        Adjacency matrix A where A[i,j] = 1 if there's an edge from i to j.

        Args:
            self_loops: If True, add self-connections (A[i,i] = 1)
                       This allows each node to retain its own features

        Returns:
            Adjacency matrix of shape (num_landmarks, num_landmarks)
        """
        adj = np.zeros((self.num_landmarks, self.num_landmarks), dtype=np.float32)

        # Add spatial edges (bidirectional - undirected graph)
        for src, dst in self.spatial_edges:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # Symmetric for undirected graph

        # Add self-loops
        if self_loops:
            adj += np.eye(self.num_landmarks, dtype=np.float32)

        return adj

    def get_normalized_adjacency(self) -> torch.Tensor:
        """
        Get normalized adjacency matrix for graph convolution.

        Normalization: D^(-1/2) * A * D^(-1/2)
        where D is the degree matrix (diagonal matrix of node degrees)

        This normalization ensures that features don't explode or vanish
        when aggregating from neighbors.

        Returns:
            Normalized adjacency matrix as torch.Tensor
        """
        adj = self.get_adjacency_matrix(self_loops=True)

        # Compute degree matrix D
        # Degree = number of connections for each node
        degree = np.sum(adj, axis=1)

        # D^(-1/2)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0  # Handle isolated nodes

        # Create diagonal matrix
        D_inv_sqrt = np.diag(degree_inv_sqrt)

        # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt

        return torch.FloatTensor(adj_normalized)

    def get_edge_index(self) -> torch.Tensor:
        """
        Get edge list in PyTorch Geometric format.

        PyTorch Geometric uses COO (coordinate) format:
        edge_index = [[source_nodes], [target_nodes]]

        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        # Include self-loops
        edges = self.spatial_edges.copy()
        for i in range(self.num_landmarks):
            edges.append((i, i))

        # Convert to PyTorch Geometric format
        edge_index = torch.LongTensor(edges).t().contiguous()

        return edge_index

    def visualize_connections(self, landmark_idx: int):
        """
        Show which landmarks are connected to a given landmark.
        Useful for debugging and understanding the graph structure.

        Args:
            landmark_idx: Index of the landmark to examine
        """
        connected = []
        for src, dst in self.spatial_edges:
            if src == landmark_idx:
                connected.append(dst)
            elif dst == landmark_idx:
                connected.append(src)

        print(f"Landmark {landmark_idx} is connected to: {sorted(set(connected))}")
        print(f"Total connections: {len(set(connected))}")


# Example usage
if __name__ == "__main__":
    # Create graph with face landmarks
    graph = LandmarkGraph(include_face=True)

    # Get adjacency matrix
    adj_matrix = graph.get_adjacency_matrix()
    print(f"\nAdjacency matrix shape: {adj_matrix.shape}")
    print(f"Sparsity: {(adj_matrix == 0).sum() / adj_matrix.size * 100:.1f}% zeros")

    # Get normalized adjacency for GCN
    adj_norm = graph.get_normalized_adjacency()
    print(f"\nNormalized adjacency shape: {adj_norm.shape}")

    # Get edge index for PyTorch Geometric
    edge_index = graph.get_edge_index()
    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Total edges (with self-loops): {edge_index.shape[1]}")

    # Visualize connections for left hand index fingertip (landmark 142 = 134 + 8)
    print("\n" + "=" * 60)
    print("Example: Left hand index fingertip connections")
    print("=" * 60)
    graph.visualize_connections(142)
