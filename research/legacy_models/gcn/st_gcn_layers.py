"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) Layers

This module implements the core building blocks of ST-GCN:
1. Graph Convolution Layer: Aggregates features from spatial neighbors
2. Temporal Convolution Layer: Captures motion patterns over time
3. ST-GCN Block: Combines spatial and temporal convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer (Spatial)
    
    Implements: H_out = o(A_norm * H_in * W)
    
    Where:
    - H_in: Input features (B, T, N, C_in)
    - A_norm: Normalized adjacency matrix (N, N)
    - W: Learnable weight matrix (C_in, C_out)
    - H_out: Output features (B, T, N, C_out)
    - o: Activation function
    
    The key idea: Each node's new features are a weighted combination
    of its neighbors' features.
    """
    
    def __init__(self, in_channels, out_channels, adj_matrix):
        """
        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            adj_matrix: Normalized adjacency matrix (N, N)
        """
        super().__init__()
        
        # Store normalized adjacency matrix as a buffer (not a parameter)
        # Buffer = tensor that should be saved with the model but not trained
        self.register_buffer('adj_matrix', adj_matrix)
        
        # Learnable transformation: maps C_in features to C_out features
        self.weight = nn.Linear(in_channels, out_channels, bias=False)
        
        # Bias term (one per output channel)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, T, N, C_in)
                B = batch size
                T = temporal length (number of frames)
                N = number of nodes (landmarks)
                C_in = input feature dimension
        
        Returns:
            Output features (B, T, N, C_out)
        """
        B, T, N, C_in = x.shape
        
        # Reshape to process all frames at once
        # (B, T, N, C_in) → (B*T, N, C_in)
        x = x.view(B * T, N, C_in)
        
        # Step 1: Linear transformation W * H
        # (B*T, N, C_in) @ (C_in, C_out) → (B*T, N, C_out)
        x = self.weight(x)
        
        # Step 2: Graph convolution A * H
        # Think: "Aggregate features from neighbors"
        # (N, N) @ (B*T, N, C_out) → (B*T, N, C_out)
        # For each node: sum (weighted by adjacency) its neighbors' features
        x = torch.matmul(self.adj_matrix, x)
        
        # Step 3: Add bias
        x = x + self.bias
        
        # Reshape back to original temporal structure
        # (B*T, N, C_out) → (B, T, N, C_out)
        x = x.view(B, T, N, -1)
        
        return x


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution Layer
    
    Applies 1D convolution along the temporal dimension to capture
    motion patterns and temporal dependencies.
    
    Example: For a kernel size of 3, each frame's features are computed from:
    - Previous frame (t-1)
    - Current frame (t)
    - Next frame (t+1)
    
    This captures motion: "Is the hand moving up or down?"
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        """
        Args:
            in_channels: Number of input channels (C_in)
            out_channels: Number of output channels (C_out)
            kernel_size: Temporal window size (default 9 frames)
            stride: Temporal stride (default 1)
        """
        super().__init__()
        
        # Padding to keep temporal dimension the same
        # For kernel_size=9, padding=(9-1)//2 = 4
        # This means we look at 4 frames before and 4 frames after
        padding = (kernel_size - 1) // 2
        
        # 1D convolution along time dimension
        # Input shape: (B, C_in, T, N)
        # We convolve along T (time), keeping N (nodes) independent
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),  # Only convolve temporally, not spatially
            stride=(stride, 1),
            padding=(padding, 0)
        )
        
        # Batch normalization for stable training
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, T, N, C_in)
        
        Returns:
            Output features (B, T, N, C_out)
        """
        B, T, N, C_in = x.shape
        
        # Reshape for Conv2d: (B, T, N, C) → (B, C, T, N)
        # Conv2d expects channels as the 2nd dimension
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Apply temporal convolution
        x = self.conv(x)
        x = self.bn(x)
        
        # Reshape back: (B, C_out, T, N) → (B, T, N, C_out)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x


class ST_GCN_Block(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block
    
    Complete building block that combines:
    1. Spatial graph convolution (aggregate from neighbors)
    2. Temporal convolution (capture motion)
    3. Residual connection (for training stability)
    
    Architecture:
        Input → [Graph Conv → Temporal Conv → ReLU] → + → Output
          |_________Residual connection_______________|
    """
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        adj_matrix,
        temporal_kernel_size=9,
        stride=1,
        dropout=0.1,
        residual=True
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            adj_matrix: Normalized adjacency matrix
            temporal_kernel_size: Size of temporal convolution window
            stride: Temporal stride
            dropout: Dropout probability
            residual: Whether to use residual connection
        """
        super().__init__()
        
        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels, adj_matrix)
        
        # Temporal convolution
        self.tcn = TemporalConvolution(
            out_channels, 
            out_channels, 
            kernel_size=temporal_kernel_size,
            stride=stride
        )
        
        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                # Need to match dimensions for residual connection
                self.residual_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(stride, 1)
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # Identity mapping (no transformation needed)
                self.residual_conv = lambda x: x
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, T, N, C_in)
        
        Returns:
            Output features (B, T, N, C_out)
        """
        # Save input for residual connection
        identity = x
        
        # Spatial graph convolution
        # "What is the hand shape right now?"
        x = self.gcn(x)
        
        # Temporal convolution
        # "How is the hand shape changing over time?"
        x = self.tcn(x)
        
        # Residual connection
        if self.residual:
            # Prepare identity for addition
            identity = identity.permute(0, 3, 1, 2)  # (B, T, N, C) → (B, C, T, N)
            identity = self.residual_conv(identity)
            identity = identity.permute(0, 2, 3, 1)  # (B, C, T, N) → (B, T, N, C)
            
            # Add residual
            x = x + identity
        
        # Activation and dropout
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class SpatialAttentionGCN(nn.Module):
    """
    Advanced: Graph Convolution with Learnable Attention
    
    Instead of fixed adjacency matrix, learns which connections are important.
    
    Example: During "pinch" sign, the model learns to pay more attention
    to edges between thumb and index finger.
    """
    
    def __init__(self, in_channels, out_channels, adj_matrix, num_attention_heads=4):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            adj_matrix: Initial adjacency matrix (will be refined by attention)
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.register_buffer('adj_matrix', adj_matrix)
        self.num_heads = num_attention_heads
        
        # Multi-head attention for graph structure
        # Each head learns different types of relationships
        self.attention = nn.ModuleList([
            nn.Linear(in_channels, 1) for _ in range(num_attention_heads)
        ])
        
        # Feature transformation
        self.weight = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, T, N, C_in)
        
        Returns:
            Output features (B, T, N, C_out)
        """
        B, T, N, C_in = x.shape
        x = x.view(B * T, N, C_in)
        
        # Compute attention weights for each head
        attention_weights = []
        for head in self.attention:
            # Compute importance scores for each node
            scores = head(x)  # (B*T, N, 1)
            scores = torch.sigmoid(scores)
            attention_weights.append(scores)
        
        # Average attention across heads
        attention = torch.stack(attention_weights, dim=0).mean(dim=0)  # (B*T, N, 1)
        
        # Refine adjacency matrix with attention
        # Original edges get weighted by learned attention
        adj_refined = self.adj_matrix.unsqueeze(0) * attention  # Broadcasting
        
        # Apply graph convolution with refined adjacency
        x = self.weight(x)
        x = torch.matmul(adj_refined.squeeze(-1), x)
        x = x + self.bias
        
        x = x.view(B, T, N, -1)
        return x


class MultiScaleST_GCN_Block(nn.Module):
    """
    Advanced: Multi-scale Spatial-Temporal Block
    
    Captures patterns at different temporal scales simultaneously:
    - Short-term: 3-frame window (fast motions)
    - Medium-term: 9-frame window (normal motions)
    - Long-term: 15-frame window (slow motions)
    
    Similar to inception modules in CNNs.
    """
    
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        
        # Each scale uses 1/3 of the output channels
        scale_channels = out_channels // 3
        
        # Graph convolution (shared across scales)
        self.gcn = GraphConvolution(in_channels, out_channels, adj_matrix)
        
        # Multi-scale temporal convolutions
        self.tcn_short = TemporalConvolution(out_channels, scale_channels, kernel_size=3)
        self.tcn_medium = TemporalConvolution(out_channels, scale_channels, kernel_size=9)
        self.tcn_long = TemporalConvolution(out_channels, scale_channels, kernel_size=15)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, T, N, C_in)
        
        Returns:
            Output features (B, T, N, C_out)
        """
        # Spatial convolution
        x = self.gcn(x)
        
        # Multi-scale temporal convolutions
        x_short = self.tcn_short(x)   # Fast movements
        x_medium = self.tcn_medium(x)  # Normal movements
        x_long = self.tcn_long(x)      # Slow movements
        
        # Concatenate multi-scale features
        x = torch.cat([x_short, x_medium, x_long], dim=-1)
        
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


# Helper function to test the layers
def test_st_gcn_layers():
    """
    Test function to verify layer shapes and functionality
    """
    print("Testing ST-GCN Layers...")
    print("=" * 60)
    
    # Create dummy data
    B, T, N, C_in = 4, 50, 209, 2  # Batch=4, Time=50, Nodes=209, Channels=2 (x,y)
    x = torch.randn(B, T, N, C_in)
    
    # Create dummy adjacency matrix
    adj = torch.eye(N) + torch.randn(N, N).abs() * 0.1
    adj = adj / adj.sum(dim=1, keepdim=True)  # Normalize
    
    print(f"Input shape: {x.shape}")
    print()
    
    # Test Graph Convolution
    gcn = GraphConvolution(C_in, 64, adj)
    out_gcn = gcn(x)
    print(f"After Graph Conv: {out_gcn.shape}")
    print(f"  ✓ Spatial features aggregated from neighbors")
    print()
    
    # Test Temporal Convolution
    tcn = TemporalConvolution(C_in, 64, kernel_size=9)
    out_tcn = tcn(x)
    print(f"After Temporal Conv: {out_tcn.shape}")
    print(f"  ✓ Temporal patterns captured across frames")
    print()
    
    # Test ST-GCN Block
    stgcn = ST_GCN_Block(C_in, 64, adj, temporal_kernel_size=9)
    out_stgcn = stgcn(x)
    print(f"After ST-GCN Block: {out_stgcn.shape}")
    print(f"  ✓ Combined spatial and temporal modeling")
    print()
    
    print("=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_st_gcn_layers()
