import json
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import os
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

# ==================== ISSUE 1: Better Feature Engineering ====================

class AdvancedFeatureExtractor:
    """Extract richer features from landmarks"""
    
    @staticmethod
    def compute_velocities(landmarks: np.ndarray) -> np.ndarray:
        """Compute frame-to-frame velocities"""
        vel = np.diff(landmarks, axis=0)
        vel = np.vstack([np.zeros_like(landmarks[:1]), vel])
        return vel
    
    @staticmethod
    def compute_accelerations(landmarks: np.ndarray) -> np.ndarray:
        """Compute accelerations"""
        vel = np.diff(landmarks, axis=0)
        acc = np.diff(vel, axis=0)
        acc = np.vstack([np.zeros_like(landmarks[:2]), acc])
        return acc
    
    @staticmethod
    def compute_angles(landmarks: np.ndarray, hand_indices: List[int]) -> np.ndarray:
        """Compute joint angles for hands"""
        angles = []
        # Hand skeleton connections
        connections = [(0,1,2), (0,5,6), (0,9,10), (0,13,14), (0,17,18)]
        
        for i, j, k in connections:
            v1 = landmarks[:, hand_indices[j], :] - landmarks[:, hand_indices[i], :]
            v2 = landmarks[:, hand_indices[k], :] - landmarks[:, hand_indices[j], :]
            
            cos_angle = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.stack(angles, axis=1)
    
    @staticmethod
    def compute_distances(landmarks: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between key landmarks"""
        # Sample key points: hands and pose
        distances = []
        pairs = [(0, 1), (1, 2), (5, 6), (11, 12)]  # Some pose pairs
        
        for i, j in pairs:
            dist = np.linalg.norm(landmarks[:, i, :] - landmarks[:, j, :], axis=1)
            distances.append(dist)
        
        return np.stack(distances, axis=1)


# ==================== ISSUE 2: Better Architecture ====================

class ImprovedTemporalCNN(nn.Module):
    """Improved CNN with residual connections and proper temporal modeling"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.conv_blocks = nn.ModuleList([
            self._build_conv_block(hidden_dim, hidden_dim, 3, dropout),
            self._build_conv_block(hidden_dim, hidden_dim, 5, dropout),
            self._build_conv_block(hidden_dim, hidden_dim, 7, dropout),
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def _build_conv_block(self, in_ch, out_ch, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x: (B, T, D)
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)   # (B, hidden_dim, T)
        
        residual = x
        for conv_block in self.conv_blocks:
            x = conv_block(x) + residual
            residual = x
        
        x = x.transpose(1, 2)   # (B, T, hidden_dim)
        x = self.output_proj(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class ImprovedTransformerEncoder(nn.Module):
    """Transformer with better initialization"""
    
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        return x


class ImprovedCNNTransformer(nn.Module):
    """Improved architecture addressing previous issues"""
    
    def __init__(
        self,
        input_dim,
        num_classes,
        cnn_hidden=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.2,
    ):
        super().__init__()
        
        # Better spatial feature extraction
        self.spatial_encoder = ImprovedTemporalCNN(
            input_dim=input_dim,
            hidden_dim=cnn_hidden,
            output_dim=d_model,
            dropout=dropout,
        )
        
        # Positional encoding with dropout
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer
        self.transformer = ImprovedTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        # Classification head with intermediate layer
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, x, mask):
        B = x.size(0)
        
        # Spatial feature extraction
        x = self.spatial_encoder(x)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Update mask for CLS token
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer encoding
        x = self.transformer(x, mask)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        return self.head(x)


# ==================== ISSUE 3: Better Training ====================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()


def mixup_batch(x, y, alpha=0.2):
    """Mixup data augmentation"""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    return mixed_x, y, y[index], lam


def train_epoch_improved(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    use_mixup: bool = True,
) -> float:
    """Improved training with mixup and better gradient handling"""
    model.train()
    train_loss = 0
    
    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        # Apply mixup
        if use_mixup:
            x, y_a, y_b, lam = mixup_batch(x, y)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(x, mask)
            
            if use_mixup:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    return train_loss / len(data_loader)


@torch.no_grad()
def test_epoch_improved(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    """Improved evaluation"""
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


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Setup
    model = ImprovedCNNTransformer(
        input_dim=400,  # Adjust based on your features
        num_classes=250,  # ASL alphabet
        cnn_hidden=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.2,
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = GradScaler(enabled=use_amp)
    
    # Use CosineAnnealingWarmRestarts for better convergence
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Training loop (pseudo-code, adapt to your data loading)
    best_accuracy = 0.0
    patience = 0
    max_patience = 20
    
    print("Training improved model...")
    print("-" * 80)
    
    for epoch in range(100):
        train_loss = train_epoch_improved(
            model, train_loader, optimizer, criterion, scaler, use_mixup=True
        )
        test_loss, test_accuracy = test_epoch_improved(model, test_loader, criterion)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience = 0
            torch.save(model.state_dict(), "best_improved_model.pth")
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("-" * 80)