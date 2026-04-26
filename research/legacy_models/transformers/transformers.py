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
        self.pe = pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class VideoTransformer(nn.Module):
    def __init__(
        self, input_dim, num_classes, d_model=192, n_heads=4, n_layers=4, dropout=0.1
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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

        x = self.input_proj(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x[:, 0])

        return self.head(x)


def train_epoch(data_loader: DataLoader) -> float:
    """
    Trains the model with the train dataloader

    Args:
        data_loader (DataLoader): DataLoader with the train frames

    Returns:
        float: Train loss for the epoch
    """
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
def test_epoch(data_loader: DataLoader) -> Tuple[float, float]:
    """
    Tests the model with the test dataloader

    Args:
        data_loader (DataLoader): DataLoader with the test frames

    Returns:
        float: Test loss for the epoch
    """
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


model = VideoTransformer(
    input_dim=2 * len(ALL_COLUMNS), num_classes=len(SIGN2INDEX_JSON)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler(enabled=use_amp)

scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)

train_loader, test_loader = get_data_loaders()

epochs = 100
least_test_loss = float("inf")
patience = 0

for epoch in range(epochs):
    train_loss = train_epoch(train_loader)
    test_loss, test_accuracy = test_epoch(test_loader)

    scheduler.step(test_loss)

    print(
        f"Epoch: {epoch} | "
        f"Train Loss: {train_loss} | "
        f"Test Loss: {test_loss} | "
        f"Test Accuracy: {test_accuracy}"
    )

    if test_loss < least_test_loss:
        least_test_loss = test_loss
        patience = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_loss": test_loss,
            },
            "best_model.pth",
        )
    else:
        patience += 1

    if patience > 15:
        break
