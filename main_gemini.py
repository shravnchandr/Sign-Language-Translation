import pandas as pd
import numpy as np
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from typing import Tuple, List, Dict

# --- Global Setup ---
# Set device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # Important: This controls the memory usage and gradient update frequency
INPUT_SIZE = 150 # Placeholder for 2 * (Number of landmarks)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Load external data (Assumes these files exist at the relative paths)
try:
    train_data = pd.read_csv("/kaggle/input/asl-signs/train.csv")
    with open("/kaggle/input/asl-signs/sign_to_prediction_index_map.json", 'r') as f:
        sign_to_prediction_index_map = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading required data files: {e}. Please check your 'Kaggle_Data' directory.")
    raise

# Pre-process train_data for fast lookup
# Create a fast lookup map: (participant_id, sequence_id) -> sign
TRAIN_MAP: Dict[Tuple[int, int], str] = train_data.set_index(['participant_id', 'sequence_id'])['sign'].to_dict()
NUM_CLASSES = len(sign_to_prediction_index_map)
LANDMARK_FILES_PATH = Path(r"/kaggle/input/asl-signs/train_landmark_files")


# --- 1. Custom PyTorch Dataset ---

class ASLLandmarkDataset(Dataset):
    """
    A PyTorch Dataset that loads a list of Parquet file paths on demand (per index).
    """
    def __init__(self, file_paths: List[Path], train_map: Dict, sign_map: Dict):
        self.file_paths = file_paths
        self.train_map = train_map
        self.sign_map = sign_map
        self.num_classes = len(sign_map)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and processes a single sequence file when requested by the DataLoader.
        Returns the feature tensor (X) and the target class index (Y).
        """
        parquet_file = self.file_paths[idx]
        
        # --- Data Loading and Processing (Optimized from previous version) ---
        try:
            # 1. Read Parquet
            landmark_data = pd.read_parquet(parquet_file)
            
            # 2. Vectorized Filtering and Imputation
            filtered_data = landmark_data[landmark_data['type'] != 'face'].fillna(0)
            
            landmark_consolidated = []
            
            # 3. Groupby Frame & Optimized Coordinate Extraction
            for _, frame_df in filtered_data.groupby("frame"):
                coordinates: List[float] = frame_df[['x', 'y']].values.flatten().tolist()
                landmark_consolidated.append(coordinates)
            
            # Convert to Tensor (Sequence length x Feature dim)
            X_tensor = torch.tensor(landmark_consolidated, dtype=torch.float32)

            # --- Label Lookup ---
            participant_id = int(parquet_file.parent.name)
            sequence_id = int(parquet_file.stem)
            signing_word = self.train_map[(participant_id, sequence_id)]
            signing_word_id = self.sign_map[signing_word]

            # Target is the class index (required by nn.CrossEntropyLoss)
            Y_tensor = torch.tensor(signing_word_id, dtype=torch.long) 
            
            return X_tensor, Y_tensor

        except Exception as e:
            # Handle corrupted files or missing labels by returning None, 
            # which will be handled by the collate_fn below.
            print(f"Error processing {parquet_file.name}. Skipping: {e}")
            return None, None


# --- 2. Custom Collate Function for Padding ---

def collate_fn_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads the sequences (X) within a batch to ensure uniform length for the LSTM.
    Filters out any None values resulting from failed __getitem__ calls.
    """
    # Filter out None values resulting from errors in __getitem__
    batch = [item for item in batch if item[0] is not None]
    
    if not batch:
        # Return empty tensors if the batch is empty
        return torch.empty(0), torch.empty(0)

    # Separate X (features) and Y (labels/indices)
    sequences, labels = zip(*batch)
    
    # 1. Find the maximum sequence length in the current batch
    max_len = max(s.size(0) for s in sequences)
    
    # 2. Pad all sequences to the max length
    padded_sequences = [
        torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)), 'constant', 0.0) 
        for s in sequences
    ]
    
    # 3. Stack the padded sequences and labels
    X_batch = torch.stack(padded_sequences)
    Y_batch = torch.stack(labels)
    
    return X_batch, Y_batch


# --- 3. Optimized Model Definition (Same as before, but crucial fix applied) ---

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # Softmax is removed because nn.CrossEntropyLoss handles it internally

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out[:, -1, :] extracts the output of the last time step for all items in the batch
        out = self.fc(out[:, -1, :])
        return out # Return raw logits


# --- 4. Main Training Setup and Loop ---

def setup_and_train():
    # 1. Gather all individual Parquet file paths
    all_file_paths = []
    # This loop assumes 'landmark_files_path' contains participant folders
    for participant_folder in LANDMARK_FILES_PATH.iterdir():
        if participant_folder.is_dir():
            for parquet_file in participant_folder.iterdir():
                if parquet_file.suffix == '.parquet':
                    # Only add the file if its sequence ID is found in the training data map
                    try:
                        p_id = int(participant_folder.name)
                        s_id = int(parquet_file.stem)
                        if (p_id, s_id) in TRAIN_MAP:
                            all_file_paths.append(parquet_file)
                    except ValueError:
                        # Skip files where participant/sequence IDs are not proper integers
                        continue

    print(f"Total sequences found: {len(all_file_paths)}")
    
    # 2. Instantiate Dataset and DataLoader
    dataset = ASLLandmarkDataset(all_file_paths, TRAIN_MAP, sign_to_prediction_index_map)
    # The DataLoader automatically handles shuffling, batching, and multi-process loading
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_pad, 
        num_workers=4, # Use multiple processes for faster data loading
        pin_memory=True if device.type == 'cuda' else False
    )

    # 3. Model, Loss, and Optimizer
    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # nn.CrossEntropyLoss expects raw logits and class indices (long tensor)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print(f"Starting training on device: {device} with batch size {BATCH_SIZE}...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            if X_batch.shape[0] == 0:
                continue # Skip empty batches
                
            # Move data to the correct device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Calculate loss
            # outputs are raw logits, Y_batch are class indices (long tensor)
            loss = criterion(outputs, Y_batch)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'\n*** Epoch {epoch+1} Finished | Average Loss: {avg_loss:.4f} ***')

# Execute the training function
setup_and_train()