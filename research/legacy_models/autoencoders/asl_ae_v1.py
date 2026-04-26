import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = r"/kaggle/input/asl-signs"
TRAIN_FILE = r"/kaggle/input/asl-signs/train.csv"

INCLUDE_FACE = False
INCLUDE_DEPTH = False

def generate_full_column_list() -> List[str]:
    """
    Generates the standardized list of 1629 column names (x/y/z for 543 landmarks).
    """
    landmark_specs = {
        'face': 468,  # Indices 0 to 467
        'left_hand': 21, # Indices 0 to 20
        'pose': 33,    # Indices 0 to 32
        'right_hand': 21, # Indices 0 to 20
    }

    if not INCLUDE_FACE:
        del landmark_specs['face']

    axes = ['x', 'y', 'z'] if INCLUDE_DEPTH else ['x', 'y']
    
    full_columns = []
    
    for landmark_type, count in landmark_specs.items():
        for i in range(count):
            for axis in axes:
                full_columns.append(f"{landmark_type}_{i}_{axis}") 
        
    return full_columns

ALL_COLUMNS = generate_full_column_list()


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize coordinates using the nose coordinates

    Args:
        df (pd.DataFrame): Unnormalaized dataframe

    Returns:
        pd.DataFrame: Normalized dataframe
    """
    axes = ['x', 'y', 'z'] if INCLUDE_DEPTH else ['x', 'y']

    origins = (
        df[(df['type'] == 'pose') & (df['landmark_index'] == 0)]
        .set_index('frame')[axes]
    )

    def normalize_frame(frame_df):
        frame = frame_df.name
        if frame not in origins.index:
            return frame_df  # or raise an error
        frame_df[axes] = frame_df[axes] - origins.loc[frame]
        return frame_df

    normalized_df = df.groupby('frame', group_keys=False).apply(normalize_frame)
    return normalized_df

def frame_stacked_data(file_path: str) -> np.ndarray:
    """
    Read landmark data from parquet files, normalize and stack them for each frame

    Args:
        file_path (str): File path for the parquet file

    Returns:
        np.ndarray: The normlaized stacked coordinates
    """
    df = pd.read_parquet(os.path.join(BASE_PATH, file_path))
    if not INCLUDE_FACE:
        df = df[df['type'] != 'face']
    
    if not INCLUDE_DEPTH:
        df.drop('z', axis=1, inplace=True)
    axes = ['x', 'y', 'z'] if INCLUDE_DEPTH else ['x', 'y']

    df['UID'] = df['type'].astype('str') + '_' + df['landmark_index'].astype('str')

    df = df.sort_values(['frame', 'UID'])
    df = normalize_values(df)

    pivot_df = df.pivot_table(index='frame', columns='UID', values=axes)
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=ALL_COLUMNS)

    final_data = pd.DataFrame(pivot_df).ffill().bfill().fillna(0).to_numpy()
    return final_data


class ASLDataset(Dataset):
    def __init__(self, coordinates_data):
        self.coordinates_data = coordinates_data

    def __len__(self):
        return len(self.coordinates_data)

    def __getitem__(self, idx):
        return torch.tensor(self.coordinates_data[idx]).float()


class ASLAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(ASLAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded


def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Reads dataloaders from the normalized coordinates

    Returns:
        Tuple[DataLoader, DataLoader]: train and test dataloaders
    """
    train_df = pd.read_csv(TRAIN_FILE)
    train_split, test_split = train_test_split(train_df, test_size=0.1, stratify=train_df['sign'])

    all_train_frames = [] 
    print("Flattening and concatenating training frames...")
    for path in train_split['path'].to_list():
        all_train_frames.extend(frame_stacked_data(path) )

    all_test_frames = []
    print("Flattening and concatenating testing frames...")
    for path in test_split['path'].to_list():
        all_test_frames.extend(frame_stacked_data(path))

    np.savez('asl_isolated.npz', train_frames=all_train_frames, test_frames=all_test_frames)

    train_dataset = ASLDataset(all_train_frames)
    test_dataset = ASLDataset(all_test_frames)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

model = ASLAutoencoder(input_size=len(ALL_COLUMNS)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs =100

train_loader, test_loader = get_data_loaders()


def train_epoch(data_loader: DataLoader) -> float:
    """
    Trains the model with the trian dataloader

    Args:
        data_loader (DataLoader): DataLoader with the train frames

    Returns:
        float: Train loss for the epoch
    """
    model.train()
    train_loss = 0

    for inputs in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(data_loader)

@torch.no_grad()
def test_epoch(data_loader: DataLoader) -> float:
    """
    Tests the model with the test dataloader

    Args:
        data_loader (DataLoader): DataLoader with the test frames

    Returns:
        float: Test loss for the epoch
    """
    model.eval()
    test_loss = 0

    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        test_loss += loss.item()

    return test_loss / len(data_loader)


least_test_loss = float('inf')
patience = 0

for epoch in range(200):
    train_loss = train_epoch(train_loader)
    test_loss = test_epoch(test_loader)
    print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")

    if test_loss < least_test_loss:
        least_test_loss = test_loss
        patience = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience += 1

    if patience == 10:
        break

