"""Face NMM (Non-Manual Markers) encoder with region-aware processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


# Face region landmark ranges (for compact 134-landmark subset)
FACE_REGIONS = {
    "nose": (0, 10),  # 10 landmarks
    "left_eye": (10, 26),  # 16 landmarks
    "right_eye": (26, 42),  # 16 landmarks
    "left_eyebrow": (42, 50),  # 8 landmarks
    "right_eyebrow": (50, 58),  # 8 landmarks
    "outer_mouth": (58, 78),  # 20 landmarks
    "inner_mouth": (78, 98),  # 20 landmarks
    "face_oval": (98, 134),  # 36 landmarks
}

# Semantic groupings for NMM features
NMM_GROUPS = {
    "eyebrows": ["left_eyebrow", "right_eyebrow"],  # Eyebrow raise/furrow
    "eyes": ["left_eye", "right_eye"],  # Eye gaze, squint, wide
    "mouth": ["outer_mouth", "inner_mouth"],  # Mouth shape, lip patterns
    "face_shape": ["nose", "face_oval"],  # Head position, tilt
}


class RegionEncoder(nn.Module):
    """Encoder for a single face region."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) region landmarks flattened

        Returns:
            (B, T, output_dim) encoded features
        """
        return self.encoder(x)


class FaceNMMEncoder(nn.Module):
    """
    Face encoder with region-specific processing for Non-Manual Markers.

    Addresses blindspot #2: Face underweighted.

    Processes face regions (eyebrows, eyes, mouth, etc.) separately
    then fuses them for better NMM representation.
    """

    def __init__(
        self,
        n_face_landmarks: int = 134,
        n_coords: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        temporal_kernel: int = 5,
    ):
        """
        Args:
            n_face_landmarks: Number of face landmarks
            n_coords: Coordinates per landmark (3 for x,y,z)
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
            temporal_kernel: Kernel size for temporal conv
        """
        super().__init__()
        self.n_face_landmarks = n_face_landmarks
        self.n_coords = n_coords
        self.output_dim = output_dim

        # Region encoders
        self.region_encoders = nn.ModuleDict()
        region_output_dim = hidden_dim // len(FACE_REGIONS)

        for region_name, (start, end) in FACE_REGIONS.items():
            n_landmarks = end - start
            input_dim = n_landmarks * n_coords
            self.region_encoders[region_name] = RegionEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=region_output_dim,
                dropout=dropout,
            )

        # Temporal modeling for each region
        self.temporal_conv = nn.ModuleDict()
        for region_name in FACE_REGIONS:
            self.temporal_conv[region_name] = nn.Sequential(
                nn.Conv1d(
                    region_output_dim,
                    region_output_dim,
                    temporal_kernel,
                    padding=temporal_kernel // 2,
                ),
                nn.BatchNorm1d(region_output_dim),
                nn.GELU(),
            )

        # NMM group fusion
        self.nmm_fusion = nn.ModuleDict()
        group_input_dim = region_output_dim * 2  # Each group has 2 regions
        for group_name in NMM_GROUPS:
            self.nmm_fusion[group_name] = nn.Sequential(
                nn.Linear(group_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim // len(NMM_GROUPS)),
            )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Attention for region importance
        self.region_attention = nn.MultiheadAttention(
            embed_dim=region_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def _extract_region(
        self, face_landmarks: torch.Tensor, region_name: str
    ) -> torch.Tensor:
        """Extract landmarks for a specific region."""
        start, end = FACE_REGIONS[region_name]
        return face_landmarks[:, :, start:end, :]

    def forward(
        self, face_landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode face landmarks with region-aware processing.

        Args:
            face_landmarks: (B, T, N_face, 3) face landmarks
            mask: Optional (B, T) validity mask

        Returns:
            (B, T, output_dim) encoded face features
        """
        B, T, N, C = face_landmarks.shape

        # Encode each region
        region_features = {}
        for region_name in FACE_REGIONS:
            # Extract region
            region = self._extract_region(face_landmarks, region_name)
            region_flat = region.reshape(B, T, -1)  # (B, T, n_landmarks * 3)

            # Encode
            encoded = self.region_encoders[region_name](
                region_flat
            )  # (B, T, region_dim)

            # Temporal conv
            encoded_t = encoded.permute(0, 2, 1)  # (B, C, T)
            encoded_t = self.temporal_conv[region_name](encoded_t)
            encoded = encoded_t.permute(0, 2, 1)  # (B, T, C)

            region_features[region_name] = encoded

        # Apply attention across regions for context
        all_regions = torch.stack(
            list(region_features.values()), dim=2
        )  # (B, T, n_regions, region_dim)
        all_regions_flat = all_regions.reshape(B * T, len(FACE_REGIONS), -1)

        attended, _ = self.region_attention(
            all_regions_flat, all_regions_flat, all_regions_flat
        )
        attended = attended.reshape(B, T, len(FACE_REGIONS), -1)

        # Update region features with attention
        for i, region_name in enumerate(FACE_REGIONS):
            region_features[region_name] = (
                region_features[region_name] + attended[:, :, i]
            )

        # Fuse NMM groups
        group_features = []
        for group_name, region_names in NMM_GROUPS.items():
            group_concat = torch.cat([region_features[r] for r in region_names], dim=-1)
            group_feat = self.nmm_fusion[group_name](group_concat)
            group_features.append(group_feat)

        # Concatenate all groups
        combined = torch.cat(group_features, dim=-1)  # (B, T, output_dim)

        # Final fusion
        output = self.final_fusion(combined)

        return output


class FaceTemporalEncoder(nn.Module):
    """
    Temporal encoder specifically for face NMM patterns.

    Captures temporal dynamics of facial expressions which are
    important for ASL grammatical markers.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-scale temporal convolutions
        self.conv_scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
                for k in [3, 5, 7]
            ]
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode face features temporally.

        Args:
            x: (B, T, input_dim) face features
            mask: Optional (B, T) validity mask

        Returns:
            (B, T, output_dim) temporally encoded features
        """
        B, T, D = x.shape

        # Apply multi-scale convolutions
        x_t = x.permute(0, 2, 1)  # (B, D, T)
        conv_outputs = [conv(x_t) for conv in self.conv_scales]

        # Concat and fuse
        concat = torch.cat(conv_outputs, dim=1)  # (B, hidden*3, T)
        concat = concat.permute(0, 2, 1)  # (B, T, hidden*3)
        fused = self.fusion(concat)  # (B, T, hidden)

        # Create attention mask if needed
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Transformer expects True for masked positions

        # Transformer
        transformed = self.transformer(fused, src_key_padding_mask=attn_mask)

        # Output
        output = self.output_proj(transformed)

        return output


class FaceChunkEncoder(nn.Module):
    """
    Chunk-based encoder for face that outputs per-chunk representations.

    Compatible with the VQ-VAE's chunked tokenization.
    """

    def __init__(
        self,
        n_face_landmarks: int = 134,
        n_coords: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 128,
        chunk_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        # Region-aware face encoder
        self.face_encoder = FaceNMMEncoder(
            n_face_landmarks=n_face_landmarks,
            n_coords=n_coords,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Temporal encoder
        self.temporal_encoder = FaceTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=2,
            dropout=dropout,
        )

        # Chunk pooling
        self.chunk_pool = nn.Sequential(
            nn.Linear(output_dim * chunk_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, face_landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode face landmarks into per-chunk representations.

        Args:
            face_landmarks: (B, T, N_face, 3)
            mask: Optional (B, T) validity mask

        Returns:
            (B, n_chunks, output_dim)
        """
        B, T, N, C = face_landmarks.shape

        # Encode with region awareness
        face_features = self.face_encoder(face_landmarks, mask)  # (B, T, hidden)

        # Temporal encoding
        temporal_features = self.temporal_encoder(
            face_features, mask
        )  # (B, T, output_dim)

        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            temporal_features = F.pad(temporal_features, (0, 0, 0, pad_len))

        T_padded = temporal_features.shape[1]
        n_chunks = T_padded // self.chunk_size

        # Reshape into chunks
        chunked = temporal_features.reshape(
            B, n_chunks, self.chunk_size, self.output_dim
        )
        chunked_flat = chunked.reshape(
            B, n_chunks, -1
        )  # (B, n_chunks, chunk_size * output_dim)

        # Pool each chunk
        output = self.chunk_pool(chunked_flat)

        return output
