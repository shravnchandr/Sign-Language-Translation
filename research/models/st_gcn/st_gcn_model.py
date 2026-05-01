"""
Complete ST-GCN Model for ASL Recognition

This implements the full model architecture that processes landmark sequences
and classifies ASL signs using spatial-temporal graph convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import our custom layers (these would be in separate files)
from research.models.st_gcn.graph_structure import LandmarkGraph
from research.models.st_gcn.st_gcn_layers import (
    ST_GCN_Block,
    GraphConvolution,
    TemporalConvolution,
)


class ST_GCN_ASL(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for ASL Recognition

    Architecture Overview:

    Input: Landmark sequences (B, T, N, C)
        B = Batch size
        T = Temporal length (number of frames)
        N = Number of landmarks (209)
        C = Coordinate dimensions (2 for x,y)

    Processing Pipeline:
    1. Input projection: Expand input features
    2. ST-GCN blocks: Extract spatial-temporal features
    3. Global pooling: Aggregate sequence information
    4. Classification: Map to sign classes

    Key Design Choices:
    - 9 ST-GCN blocks organized in 3 stages (like ResNet)
    - Gradual channel increase: 64 → 128 → 256
    - Temporal pooling to reduce sequence length
    - Global average pooling for final representation
    """

    def __init__(
        self,
        num_classes,
        adj_matrix,
        in_channels=2,  # x, y coordinates
        num_landmarks=209,
        temporal_kernel_size=9,
        dropout=0.1,
        edge_importance_weighting=True,
    ):
        """
        Args:
            num_classes: Number of ASL sign classes
            adj_matrix: Normalized adjacency matrix (N, N)
            in_channels: Input feature dimension (2 for x,y or 3 for x,y,z)
            num_landmarks: Number of body landmarks (209 with face, 75 without)
            temporal_kernel_size: Temporal convolution window size
            dropout: Dropout probability
            edge_importance_weighting: If True, learn importance of each edge
        """
        super().__init__()

        self.num_landmarks = num_landmarks

        # Register adjacency matrix
        self.register_buffer("adj_matrix", adj_matrix)

        # Stage 0: Input projection
        # Project 2D coordinates to higher-dimensional feature space
        # This gives the model more representational power
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: Low-level spatial-temporal features (64 channels)
        # Learns: "What are the basic hand shapes and movements?"
        self.stage1 = nn.ModuleList(
            [
                ST_GCN_Block(
                    64, 64, adj_matrix, temporal_kernel_size, stride=1, dropout=dropout
                ),
                ST_GCN_Block(
                    64, 64, adj_matrix, temporal_kernel_size, stride=1, dropout=dropout
                ),
                ST_GCN_Block(
                    64, 128, adj_matrix, temporal_kernel_size, stride=2, dropout=dropout
                ),
            ]
        )

        # Stage 2: Mid-level features (128 channels)
        # Learns: "How do hand shapes combine and transition?"
        self.stage2 = nn.ModuleList(
            [
                ST_GCN_Block(
                    128,
                    128,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    128,
                    128,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    128,
                    256,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=2,
                    dropout=dropout,
                ),
            ]
        )

        # Stage 3: High-level features (256 channels)
        # Learns: "What is the complete sign pattern?"
        self.stage3 = nn.ModuleList(
            [
                ST_GCN_Block(
                    256,
                    256,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    256,
                    256,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    256,
                    256,
                    adj_matrix,
                    temporal_kernel_size,
                    stride=1,
                    dropout=dropout,
                ),
            ]
        )

        # Edge importance weighting (optional advanced feature)
        # Allows model to learn which skeletal connections are most important
        self.edge_importance_weighting = edge_importance_weighting
        if edge_importance_weighting:
            # Learnable weight for each edge
            # Initially all 1.0, but model can adjust based on task
            num_edges = (adj_matrix > 0).sum().item()
            self.edge_importance = nn.Parameter(torch.ones(num_edges))

        # Global pooling
        # Aggregate information across time and space
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

        print(f"ST-GCN Model created:")
        print(f"  Input: (B, {in_channels}, T, {num_landmarks})")
        print(f"  Stages: 3 (64→128→256 channels)")
        print(f"  Total ST-GCN blocks: 9")
        print(f"  Output classes: {num_classes}")

    def forward(self, x, mask=None):
        """
        Forward pass through the network

        Args:
            x: Input landmarks (B, T, N, C) where C=2 (x,y coordinates)
            mask: Optional attention mask (B, T) for variable-length sequences

        Returns:
            logits: Class predictions (B, num_classes)
        """
        B, T, N, C = x.shape

        # Reshape for Conv2d processing: (B, T, N, C) → (B, C, T, N)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Input projection: (B, 2, T, N) → (B, 64, T, N)
        x = self.input_projection(x)

        # Reshape for ST-GCN blocks: (B, C, T, N) → (B, T, N, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Stage 1: Low-level features
        for block in self.stage1:
            x = block(x)

        # Stage 2: Mid-level features
        for block in self.stage2:
            x = block(x)

        # Stage 3: High-level features
        for block in self.stage3:
            x = block(x)

        # x shape now: (B, T', N, 256) where T' is reduced by strides

        # Apply mask if provided (for variable-length sequences)
        if mask is not None:
            # Downsample mask to match current temporal resolution
            stride_factor = T // x.shape[1]
            mask = mask[:, ::stride_factor]
            mask = mask.unsqueeze(2).unsqueeze(3)  # (B, T', 1, 1)
            x = x * mask

        # Reshape for global pooling: (B, T', N, 256) → (B, 256, T', N)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Global average pooling: (B, 256, T', N) → (B, 256, 1, 1)
        x = self.global_pool(x)

        # Flatten: (B, 256, 1, 1) → (B, 256)
        x = x.view(B, -1)

        # Classification: (B, 256) → (B, num_classes)
        logits = self.classifier(x)

        return logits

    def extract_features(self, x, mask=None):
        """
        Extract feature embeddings without classification.
        Useful for visualization or transfer learning.

        Args:
            x: Input landmarks (B, T, N, C)
            mask: Optional mask (B, T)

        Returns:
            features: Feature embeddings (B, 256)
        """
        B, T, N, C = x.shape

        # Forward pass up to global pooling
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.input_projection(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        for block in self.stage1:
            x = block(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)

        if mask is not None:
            stride_factor = T // x.shape[1]
            mask = mask[:, ::stride_factor]
            mask = mask.unsqueeze(2).unsqueeze(3)
            x = x * mask

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.global_pool(x)
        features = x.view(B, -1)

        return features


class LightweightST_GCN(nn.Module):
    """
    Lightweight version of ST-GCN for faster training and inference.

    Differences from full model:
    - Fewer blocks (6 instead of 9)
    - Smaller channel dimensions (32→64→128 instead of 64→128→256)
    - Suitable for quick experiments or resource-constrained scenarios
    """

    def __init__(
        self, num_classes, adj_matrix, in_channels=2, num_landmarks=209, dropout=0.1
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.register_buffer("adj_matrix", adj_matrix)

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Simplified architecture
        self.layers = nn.ModuleList(
            [
                ST_GCN_Block(
                    32, 32, adj_matrix, temporal_kernel_size=9, dropout=dropout
                ),
                ST_GCN_Block(
                    32,
                    64,
                    adj_matrix,
                    temporal_kernel_size=9,
                    stride=2,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    64, 64, adj_matrix, temporal_kernel_size=9, dropout=dropout
                ),
                ST_GCN_Block(
                    64,
                    128,
                    adj_matrix,
                    temporal_kernel_size=9,
                    stride=2,
                    dropout=dropout,
                ),
                ST_GCN_Block(
                    128, 128, adj_matrix, temporal_kernel_size=9, dropout=dropout
                ),
                ST_GCN_Block(
                    128, 128, adj_matrix, temporal_kernel_size=9, dropout=dropout
                ),
            ]
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

        print(f"Lightweight ST-GCN Model created:")
        print(f"  Channels: 32→64→128")
        print(f"  Blocks: 6")
        print(f"  Parameters: ~{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, x, mask=None):
        B, T, N, C = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.input_projection(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        if mask is not None:
            stride_factor = T // x.shape[1]
            mask = mask[:, ::stride_factor]
            mask = mask.unsqueeze(2).unsqueeze(3)
            x = x * mask

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.global_pool(x)
        x = x.view(B, -1)
        logits = self.classifier(x)

        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model with dummy data"""
    print("Testing ST-GCN Model...")
    print("=" * 60)

    # Dummy data
    B, T, N, C = 4, 100, 209, 2
    x = torch.randn(B, T, N, C)
    mask = torch.ones(B, T, dtype=torch.bool)
    num_classes = 250

    # Create dummy adjacency matrix
    adj = torch.eye(N) + torch.randn(N, N).abs() * 0.1
    adj = adj / adj.sum(dim=1, keepdim=True)

    # Test full model
    print("1. Full ST-GCN Model:")
    model_full = ST_GCN_ASL(num_classes, adj)
    logits = model_full(x, mask)
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Parameters: {count_parameters(model_full):,}")
    print()

    # Test lightweight model
    print("2. Lightweight ST-GCN Model:")
    model_light = LightweightST_GCN(num_classes, adj)
    logits = model_light(x, mask)
    print(f"   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Parameters: {count_parameters(model_light):,}")
    print()

    # Test feature extraction
    print("3. Feature Extraction:")
    features = model_full.extract_features(x, mask)
    print(f"   Features: {features.shape}")
    print()

    print("=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_model()
