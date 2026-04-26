import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Tuple, List, Dict, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
HIDDEN_SIZE = 128  # Increased for better capacity
NUM_LAYERS = 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT = 0.3  # Added dropout for regularization
PATIENCE = 5  # Early stopping patience


class ASLLandmarkDataset(Dataset):
    def __init__(self, features, labels, sign_to_index_map):
        self.features = features
        self.labels = labels
        self.sign_to_index_map = sign_to_index_map

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        X_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        Y_tensor = torch.tensor(self.sign_to_index_map[self.labels[idx]], dtype=torch.long)

        return X_tensor, Y_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout before final layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Optional: Use pack_padded_sequence for efficiency
        if lengths is not None:
            # Sort by length (required by pack_padded_sequence)
            lengths_sorted, perm_idx = lengths.sort(descending=True)
            x_sorted = x[perm_idx]
            
            # Pack the padded sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True
            )
            packed_out, (hn, cn) = self.lstm(packed, (h0, c0))
            
            # Restore original order
            _, unperm_idx = perm_idx.sort()
            hn = hn[:, unperm_idx, :]
            
            # Use final hidden state from last layer
            out = hn[-1]
        else:
            # Standard forward pass
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
            out = lstm_out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc(out)
        return out

def collate_fn_pad(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads sequences and returns sequence lengths for pack_padded_sequence.
    """
    # Filter out None values
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    sequences, labels = zip(*batch)
    
    # Get sequence lengths before padding
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    
    # Find max length and pad
    max_len = lengths.max().item()
    padded_sequences = [
        torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)), 'constant', 0.0) 
        for s in sequences
    ]
    
    X_batch = torch.stack(padded_sequences)
    Y_batch = torch.stack(labels)
    
    return X_batch, Y_batch, lengths


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch_idx, (X_batch, Y_batch, lengths) in enumerate(dataloader):    
        if X_batch.shape[0] == 0:
            continue
            
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, Y_batch)
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += calculate_accuracy(outputs, Y_batch)
        num_batches += 1

        # Print progress every 50 batches
        # if (batch_idx + 1) % 50 == 0:
        #     print(f"  Train Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch, lengths) in enumerate(dataloader): 
            if X_batch.shape[0] == 0:
                continue
                
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            lengths = lengths.to(device)
            
            outputs = model(X_batch, lengths)
            loss = criterion(outputs, Y_batch)
            
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, Y_batch)
            num_batches += 1

            # Print progress every 50 batches
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"  Val Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc


with open("/Users/shravnchandr/Downloads/isolated_asl_data.pkl", "rb") as f:
    features, labels = pickle.load(f)

with open("/kaggle/input/asl-signs/sign_to_prediction_index_map.json", 'r') as f:
    sign_to_index_map = json.load(f)

NUM_CLASSES = len(sign_to_index_map)

# Train/validation split with stratification
train_features, val_features, train_labels, val_labels = train_test_split(
    features, labels, test_size=0.15, random_state=42, stratify=labels
)

# Determine INPUT_SIZE from first sample
temp_dataset = ASLLandmarkDataset([train_features[0]], [labels[0]], sign_to_index_map)
sample_x, _ = temp_dataset[0]
INPUT_SIZE = sample_x.shape[1]
print(f"Detected INPUT_SIZE: {INPUT_SIZE}")

# Create datasets and dataloaders
train_dataset = ASLLandmarkDataset(train_features, train_labels, sign_to_index_map)
val_dataset = ASLLandmarkDataset(val_features, val_labels, sign_to_index_map)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn_pad, 
    num_workers=2,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_pad,
    num_workers=2,
    pin_memory=True if device.type == 'cuda' else False
)

# Model, optimizer, scheduler
model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()


# Training loop with early stopping
print(f"\nStarting training on {device}...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, 'best_model.pth')
        print(f'✓ Best model saved!')
    else:
        patience_counter += 1
        
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping triggered after {epoch+1} epochs')
        break