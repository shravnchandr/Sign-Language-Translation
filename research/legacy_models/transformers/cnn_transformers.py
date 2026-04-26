import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from data_prep import ALL_COLUMNS, SIGN2INDEX_JSON, get_data_loaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TemporalCNN(nn.Module):
    """
    1D CNN for extracting spatial features from landmarks at each timestep
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        # 1D convolutions process each frame's landmarks
        # We use kernel_size=1 to process spatial relationships without mixing timesteps
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, T, D) where D is input_dim
        # Transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)

        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        # Transpose back to (B, T, output_dim)
        x = x.transpose(1, 2)

        return x


class SpatialFeatureExtractor(nn.Module):
    """
    Alternative: Uses MLPs with residual connections to extract spatial features
    More flexible than CNN for sparse landmark data
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Residual connection
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        # x shape: (B, T, D)
        identity = self.residual_proj(x)

        out = self.mlp1(x)
        out = self.mlp2(out)
        out = self.mlp3(out)

        return out + identity


class CNNTransformerModel(nn.Module):
    """
    Hybrid CNN + Transformer model for ASL recognition

    Architecture:
    1. Spatial Feature Extraction: CNN/MLP processes landmarks at each frame
    2. Temporal Modeling: Transformer captures motion patterns across time
    3. Classification: CLS token pooling + linear head
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        cnn_hidden=256,
        d_model=192,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        use_cnn=True,  # Set False to use MLP instead
    ):
        super().__init__()

        # Spatial feature extractor
        if use_cnn:
            self.spatial_encoder = TemporalCNN(
                input_dim=input_dim,
                hidden_dim=cnn_hidden,
                output_dim=d_model,
                dropout=dropout,
            )
        else:
            self.spatial_encoder = SpatialFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=cnn_hidden,
                output_dim=d_model,
                dropout=dropout,
            )

        # Positional encoding for temporal information
        self.pos_enc = PositionalEncoding(d_model)

        # CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        B, T, _ = x.shape

        # Step 1: Extract spatial features at each timestep
        x = self.spatial_encoder(x)  # (B, T, d_model)

        # Step 2: Add positional encoding
        x = self.pos_enc(x)

        # Step 3: Add CLS token
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        # Update mask for CLS token
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Step 4: Transformer encoding for temporal modeling
        x = self.encoder(x, src_key_padding_mask=~mask)

        # Step 5: Classification from CLS token
        x = self.norm(x[:, 0])

        return self.head(x)


class MultiScaleCNNTransformer(nn.Module):
    """
    Advanced version: Multi-scale temporal convolutions + Transformer
    Captures patterns at different temporal scales
    """

    def __init__(
        self, input_dim, num_classes, d_model=192, n_heads=4, n_layers=4, dropout=0.1
    ):
        super().__init__()

        # Multi-scale spatial feature extraction
        self.conv_3 = nn.Conv1d(input_dim, d_model // 3, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(input_dim, d_model // 3, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(input_dim, d_model // 3, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        B, T, _ = x.shape

        # Multi-scale feature extraction
        x_t = x.transpose(1, 2)  # (B, D, T)

        x3 = self.conv_3(x_t)
        x5 = self.conv_5(x_t)
        x7 = self.conv_7(x_t)

        # Concatenate multi-scale features
        x = torch.cat([x3, x5, x7], dim=1)  # (B, d_model, T)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Add CLS token
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x[:, 0])

        return self.head(x)


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> float:
    """Train for one epoch"""
    model.train()
    train_loss = 0

    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(x, mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(data_loader)


@torch.no_grad()
def test_epoch(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    correct, total = 0, 0

    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        logits = model(x, mask)

        loss = criterion(logits, y)
        test_loss += loss.item()

        prediction = logits.argmax(dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)

    return test_loss / len(data_loader), correct / total


# Choose which model to use
MODEL_TYPE = (
    "cnn_transformer"  # Options: "cnn_transformer", "mlp_transformer", "multiscale"
)

# if MODEL_TYPE == "cnn_transformer":
#     model = CNNTransformerModel(
#         input_dim=2 * len(ALL_COLUMNS),
#         num_classes=len(SIGN2INDEX_JSON),
#         cnn_hidden=256,
#         d_model=192,
#         n_heads=4,
#         n_layers=4,
#         dropout=0.1,
#         use_cnn=True,
#     ).to(device)
if MODEL_TYPE == "mlp_transformer":
    model = CNNTransformerModel(
        input_dim=2 * len(ALL_COLUMNS),
        num_classes=len(SIGN2INDEX_JSON),
        cnn_hidden=256,
        d_model=192,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        use_cnn=False,  # Use MLP instead of CNN
    ).to(device)
elif MODEL_TYPE == "multiscale":
    model = MultiScaleCNNTransformer(
        input_dim=2 * len(ALL_COLUMNS),
        num_classes=len(SIGN2INDEX_JSON),
        d_model=192,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)
else:
    model = CNNTransformerModel(
        input_dim=2 * len(ALL_COLUMNS),
        num_classes=len(SIGN2INDEX_JSON),
        cnn_hidden=256,
        d_model=192,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        use_cnn=True,
    ).to(device)

print(f"Model: {MODEL_TYPE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler(enabled=use_amp)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)

train_loader, test_loader = get_data_loaders()

epochs = 100
best_test_loss = float("inf")
best_test_accuracy = 0.0
patience = 0
max_patience = 15

print(f"Training on {device}")
print("-" * 80)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
    test_loss, test_accuracy = test_epoch(model, test_loader, criterion)

    # Step scheduler
    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_test_accuracy = test_accuracy
        patience = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            },
            "best_model.pth",
        )
        print(f"  → Saved best model")
    else:
        patience += 1

    if patience >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print("-" * 80)
print("Training completed!")
print(f"Best test loss: {best_test_loss:.4f}")
print(f"Best test accuracy: {best_test_accuracy:.4f}")
