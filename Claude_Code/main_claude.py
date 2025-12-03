import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Tuple, List, Dict, Optional

# --- Global Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HIDDEN_SIZE = 128  # Increased for better capacity
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT = 0.3  # Added dropout for regularization
PATIENCE = 5  # Early stopping patience

# Load external data
try:
    train_data = pd.read_csv("/kaggle/input/asl-signs/train.csv")
    with open("/kaggle/input/asl-signs/sign_to_prediction_index_map.json", 'r') as f:
        sign_to_prediction_index_map = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading required data files: {e}")
    raise

# Pre-process train_data for fast lookup
TRAIN_MAP: Dict[Tuple[int, int], str] = train_data.set_index(['participant_id', 'sequence_id'])['sign'].to_dict()
NUM_CLASSES = len(sign_to_prediction_index_map)
LANDMARK_FILES_PATH = Path(r"/kaggle/input/asl-signs/train_landmark_files")


# --- 1. Custom PyTorch Dataset ---

class ASLLandmarkDataset(Dataset):
    def __init__(self, file_paths: List[Path], train_map: Dict, sign_map: Dict):
        self.file_paths = file_paths
        self.train_map = train_map
        self.sign_map = sign_map
        self.num_classes = len(sign_map)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        parquet_file = self.file_paths[idx]
        
        try:
            # Read and filter data
            landmark_data = pd.read_parquet(parquet_file)
            filtered_data = landmark_data[landmark_data['type'] != 'face'].fillna(0)
            
            # Extract coordinates per frame (vectorized)
            landmark_consolidated = []
            for _, frame_df in filtered_data.groupby("frame"):
                coordinates = frame_df[['x', 'y']].values.flatten()
                landmark_consolidated.append(coordinates)
            
            if not landmark_consolidated:
                return None, None
                
            X_tensor = torch.tensor(np.array(landmark_consolidated), dtype=torch.float32)
            
            # Get label
            participant_id = int(parquet_file.parent.name)
            sequence_id = int(parquet_file.stem)
            signing_word = self.train_map.get((participant_id, sequence_id))
            
            if signing_word is None:
                return None, None
                
            signing_word_id = self.sign_map[signing_word]
            Y_tensor = torch.tensor(signing_word_id, dtype=torch.long)
            
            return X_tensor, Y_tensor

        except Exception as e:
            print(f"Error processing {parquet_file.name}: {e}")
            return None, None


# --- 2. Custom Collate Function ---

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


# --- 3. Improved LSTM Model ---

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


# --- 4. Training and Evaluation ---

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
    
    for X_batch, Y_batch, lengths in dataloader:
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
        for X_batch, Y_batch, lengths in dataloader:
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
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc


def setup_and_train():
    # 1. Gather all file paths
    all_file_paths = []
    for participant_folder in LANDMARK_FILES_PATH.iterdir():
        if participant_folder.is_dir():
            for parquet_file in participant_folder.iterdir():
                if parquet_file.suffix == '.parquet':
                    try:
                        p_id = int(participant_folder.name)
                        s_id = int(parquet_file.stem)
                        if (p_id, s_id) in TRAIN_MAP:
                            all_file_paths.append(parquet_file)
                    except ValueError:
                        continue

    print(f"Total sequences found: {len(all_file_paths)}")
    
    # 2. Train/validation split
    train_paths, val_paths = train_test_split(
        all_file_paths, test_size=0.15, random_state=42
    )
    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    
    # 3. Determine INPUT_SIZE from first sample
    temp_dataset = ASLLandmarkDataset([train_paths[0]], TRAIN_MAP, sign_to_prediction_index_map)
    sample_x, _ = temp_dataset[0]
    INPUT_SIZE = sample_x.shape[1]
    print(f"Detected INPUT_SIZE: {INPUT_SIZE}")
    
    # 4. Create datasets and dataloaders
    train_dataset = ASLLandmarkDataset(train_paths, TRAIN_MAP, sign_to_prediction_index_map)
    val_dataset = ASLLandmarkDataset(val_paths, TRAIN_MAP, sign_to_prediction_index_map)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_pad, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_pad,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 5. Model, optimizer, scheduler
    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training loop with early stopping
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
    
    print('\nTraining completed!')

# Execute training
if __name__ == "__main__":
    setup_and_train()