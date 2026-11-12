import pandas as pd
import numpy as np
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

# --- Global Setup ---

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load external data (Assumes these files exist at the relative paths)
try:
    train_data = pd.read_csv("Kaggle_Data/train.csv")
    with open("Kaggle_Data/sign_to_prediction_index_map.json", 'r') as f:
        sign_to_prediction_index_map = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading required data files: {e}. Please check your 'Kaggle_Data' directory.")
    raise

# Pre-process train_data for fast lookup (Optimization)
# Create a fast lookup map: (participant_id, sequence_id) -> sign
TRAIN_MAP = train_data.set_index(['participant_id', 'sequence_id'])['sign'].to_dict()
NUM_CLASSES = len(sign_to_prediction_index_map)


# --- Optimized Data Aggregation Function ---

def aggregate_data_optimized(
    folder_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates landmark data from Parquet files in a directory, 
    applying optimizations for speed and correctness.
    """
    X_data, y_data = [], []

    # Iterate over all files in the given folder (participant ID folder)
    for parquet_file in folder_path.iterdir():
        if parquet_file.suffix != '.parquet':
            continue

        try:
            # 1. CRITICAL FIX: Read the correct file from the iteration
            landmark_data = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"Error reading Parquet file {parquet_file.name}: {e}. Skipping.")
            continue
        
        # 2. Vectorized Filtering and Imputation (Optimization)
        # Filter out 'face' landmarks and fill NaNs with 0 in one operation
        filtered_data = landmark_data[landmark_data['type'] != 'face'].fillna(0)
        
        landmark_consolidated = []
        
        # 3. Groupby Frame & Optimized Coordinate Extraction (Optimization)
        for _, frame_df in filtered_data.groupby("frame"):
            # Select x and y columns and flatten them into a single list of coordinates
            # [x1, y1, x2, y2, ...] for all landmarks in the frame.
            coordinates: List[float] = frame_df[['x', 'y']].values.flatten().tolist()
            landmark_consolidated.append(coordinates)
        
        # --- Label Lookup (Optimization) ---
        
        # 4. Optimized Label Lookup using the pre-computed map
        try:
            participant_id = int(parquet_file.parent.name)
            sequence_id = int(parquet_file.stem)
            
            # O(1) dictionary lookup
            signing_word = TRAIN_MAP[(participant_id, sequence_id)]
            signing_word_id = sign_to_prediction_index_map[signing_word]

        except (KeyError, ValueError) as e:
            # Catch cases where ID parsing fails or the sequence is missing in train_data
            print(f"Warning: Skipping {parquet_file.name}. Could not find label: {e}")
            continue

        X_data.append(landmark_consolidated)
        y_data.append(signing_word_id)

    if not y_data:
        # Return empty tensors if no data was processed
        return torch.empty(0), torch.empty(0)

    # 5. Efficient One-Hot Encoding
    one_hot_matrix = np.zeros((len(y_data), NUM_CLASSES))
    one_hot_matrix[np.arange(len(y_data)), y_data] = 1

    # Convert to Tensors and move to device
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(one_hot_matrix, dtype=torch.float32).to(device)

    return X_tensor, Y_tensor

# --- Optimized Model Definition ---

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input_size should be 2 * (Number of non-face landmarks)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # 6. OPTIMIZATION: Removed self.softmax (CrossEntropyLoss handles it)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the last time step output for classification
        out = self.fc(out[:, -1, :])
        
        # Return raw logits (required by nn.CrossEntropyLoss)
        return out


# --- Training Setup ---

# Check your data format. If you have 25 hands landmarks and 17 pose landmarks, 
# input_size = (25 + 17) * 2 = 84. The original 150 seems arbitrary; verify this value.
input_size = 150 
hidden_size = 64
num_layers = 2
num_classes = NUM_CLASSES

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model.to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# NOTE: CrossEntropyLoss expects target (Y) to be class indices, not one-hot vectors.
# Since your data processing produces a one-hot matrix, we must use a workaround:
# Use torch.argmax(Y_tensor, dim=1) to get the class indices.
criterion = nn.CrossEntropyLoss()

# The path to the main directory containing participant folders
landmark_files_path = Path(
    r"/kaggle/input/asl-signs/train_landmark_files"
)
num_epochs = 50


# --- Optimized Training Loop (Iterating over files/participants) ---

# We will iterate over files sequentially for simplicity, but a DataLoader is better.
# For demonstration, the data loading is still per-participant, but using the optimized function.
print(f"Starting training on device: {device}")

for folder_path in landmark_files_path.iterdir():
    # Only process directories (participant folders)
    if not folder_path.is_dir():
        continue
        
    print(f"\n--- Processing Participant Folder: {folder_path.name} ---")
    
    # Load all sequences (videos) for the current participant
    X_tensor, Y_tensor_onehot = aggregate_data_optimized(folder_path)
    
    if X_tensor.shape[0] == 0:
        print("No data loaded for this participant. Skipping.")
        continue

    # Convert one-hot target to class indices required by nn.CrossEntropyLoss (Fix for Loss)
    Y_tensor_indices = torch.argmax(Y_tensor_onehot, dim=1).long()

    # Training on this batch of data (all sequences from one participant)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_tensor)
        
        # Calculate loss using the indices
        loss = criterion(outputs, Y_tensor_indices)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Optional: Print loss less frequently
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')