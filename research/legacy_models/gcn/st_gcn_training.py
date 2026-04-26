"""
Complete Training Script for ST-GCN ASL Recognition

This script ties everything together:
1. Loads your existing data from data_prep.py
2. Creates the graph structure
3. Trains the ST-GCN model
4. Evaluates and saves the best model

To use: Simply run `python st_gcn_training.py`
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from typing import Tuple
import numpy as np

# Import your existing data loading
from data_prep import get_data_loaders, ALL_COLUMNS, SIGN2INDEX_JSON

# Import ST-GCN components (these would be in separate files)
from graph_structure import LandmarkGraph
from st_gcn_model import ST_GCN_ASL, LightweightST_GCN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


def adapt_dataloader_for_stgcn(train_loader, test_loader):
    """
    Your existing data loader returns (x, mask, y) where:
    - x: (B, T, D) with D = 418 (209 landmarks × 2 coords for pos+vel)
    - mask: (B, T)
    - y: (B,)
    
    ST-GCN expects:
    - x: (B, T, N, C) where N=209 landmarks, C=2 coords (just position, not velocity)
    - mask: (B, T) - same
    - y: (B,) - same
    
    This wrapper extracts just the position features (first 209×2=418 features).
    """
    
    class ST_GCN_DataLoaderWrapper:
        """Wrapper to reshape data for ST-GCN"""
        
        def __init__(self, dataloader, num_landmarks=209):
            self.dataloader = dataloader
            self.num_landmarks = num_landmarks
        
        def __iter__(self):
            for x, mask, y in self.dataloader:
                B, T, D = x.shape
                # D = 418 (209 landmarks × 2 coords × 2 for pos+vel)
                # We want just position: first 418 features
                # Split: position (209×2=418) and velocity (209×2=418)
                
                coords_per_landmark = 2  # x, y (no z in your data)
                total_features = self.num_landmarks * coords_per_landmark
                
                # Extract position only (first half of features)
                x_pos = x[:, :, :total_features]  # (B, T, 418)
                
                # Reshape to (B, T, N, C)
                x_reshaped = x_pos.reshape(B, T, self.num_landmarks, coords_per_landmark)
                
                yield x_reshaped, mask, y
        
        def __len__(self):
            return len(self.dataloader)
    
    return ST_GCN_DataLoaderWrapper(train_loader), ST_GCN_DataLoaderWrapper(test_loader)


def train_epoch(
    model: nn.Module,
    data_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> float:
    """
    Train for one epoch
    
    Args:
        model: ST-GCN model
        data_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scaler: Gradient scaler for mixed precision
    
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0
    num_batches = 0
    
    for x, mask, y in data_loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(x, mask)
            loss = criterion(logits, y)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        num_batches += 1
    
    return train_loss / num_batches


@torch.no_grad()
def test_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate on test set
    
    Args:
        model: ST-GCN model
        data_loader: Test data loader
        criterion: Loss function
    
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    for x, mask, y in data_loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        # Forward pass
        logits = model(x, mask)
        loss = criterion(logits, y)
        
        # Compute accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == y).sum().item()
        total += y.size(0)
        
        test_loss += loss.item()
        num_batches += 1
    
    return test_loss / num_batches, correct / total


def main():
    """
    Main training loop
    """
    print("="*80)
    print("ST-GCN ASL Recognition Training")
    print("="*80)
    
    # ==========================
    # 1. Load Data
    # ==========================
    print("\n[1/5] Loading data...")
    train_loader, test_loader = get_data_loaders()
    
    # Wrap data loaders for ST-GCN format
    num_landmarks = len(ALL_COLUMNS) // 2  # Divide by 2 because we have x,y coords
    print(f"Number of landmarks: {num_landmarks}")
    
    train_loader_stgcn, test_loader_stgcn = adapt_dataloader_for_stgcn(
        train_loader, test_loader
    )
    
    # ==========================
    # 2. Create Graph Structure
    # ==========================
    print("\n[2/5] Creating graph structure...")
    
    # NOTE: This assumes you have LandmarkGraph class available
    # from graph_structure import LandmarkGraph
    
    # For now, we'll create a simple adjacency based on your landmark ordering
    # You should replace this with the proper LandmarkGraph from the artifacts
    
    include_face = num_landmarks > 100  # 209 with face, 75 without
    
    # Create simple adjacency (you should use LandmarkGraph instead)
    # adj = torch.eye(num_landmarks, dtype=torch.float32)
    
    # Add some connectivity (this is simplified - use proper graph structure!)
    # In practice, use: graph = LandmarkGraph(include_face=include_face)
    #                   adj = graph.get_normalized_adjacency()
    
    graph = LandmarkGraph(include_face=include_face)
    adj = graph.get_normalized_adjacency()
    
    # For demonstration, add nearest neighbor connections
    for i in range(num_landmarks - 1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
    
    # Normalize adjacency
    degree = adj.sum(dim=1, keepdim=True)
    adj = adj / degree
    
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Graph sparsity: {(adj == 0).sum().item() / adj.numel() * 100:.1f}% zeros")
    
    # ==========================
    # 3. Create Model
    # ==========================
    print("\n[3/5] Creating ST-GCN model...")
    
    num_classes = len(SIGN2INDEX_JSON)
    
    # Choose model type
    MODEL_TYPE = "full"  # Options: "full", "lightweight"
    
    if MODEL_TYPE == "full":
        model = ST_GCN_ASL(
            num_classes=num_classes,
            adj_matrix=adj,
            in_channels=2,  # x, y coordinates
            num_landmarks=num_landmarks,
            temporal_kernel_size=9,
            dropout=0.1
        ).to(device)
    else:
        model = LightweightST_GCN(
            num_classes=num_classes,
            adj_matrix=adj,
            in_channels=2,
            num_landmarks=num_landmarks,
            dropout=0.1
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {MODEL_TYPE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==========================
    # 4. Setup Training
    # ==========================
    print("\n[4/5] Setting up training...")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Slightly higher LR for ST-GCN
        weight_decay=0.01
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Mixed precision training
    scaler = GradScaler(enabled=use_amp)
    
    # Training settings
    epochs = 100
    best_test_loss = float('inf')
    best_test_accuracy = 0.0
    patience = 0
    max_patience = 20
    
    print(f"Training on: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Max epochs: {epochs}")
    print(f"Early stopping patience: {max_patience}")
    
    # ==========================
    # 5. Training Loop
    # ==========================
    print("\n[5/5] Starting training...")
    print("="*80)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader_stgcn, optimizer, criterion, scaler)
        
        # Evaluate
        test_loss, test_accuracy = test_epoch(model, test_loader_stgcn, criterion)
        
        # Step scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_accuracy = test_accuracy
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'num_landmarks': num_landmarks,
                'num_classes': num_classes,
            }, 'best_st_gcn_model.pth')
            
            print(f"  → Saved best model (acc: {test_accuracy:.4f})")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # ==========================
    # Final Results
    # ==========================
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print("="*80)
    
    return model, best_test_loss, best_test_accuracy


def load_and_evaluate(model_path='best_st_gcn_model.pth'):
    """
    Load a saved model and evaluate it
    
    Args:
        model_path: Path to saved model checkpoint
    """
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    num_landmarks = checkpoint['num_landmarks']
    num_classes = checkpoint['num_classes']
    
    # Create adjacency (simplified - use proper graph in practice)
    adj = torch.eye(num_landmarks, dtype=torch.float32)
    for i in range(num_landmarks - 1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
    degree = adj.sum(dim=1, keepdim=True)
    adj = adj / degree
    
    # Create model
    model = ST_GCN_ASL(
        num_classes=num_classes,
        adj_matrix=adj,
        in_channels=2,
        num_landmarks=num_landmarks,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Test accuracy: {checkpoint['test_accuracy']:.4f}")
    
    return model


if __name__ == "__main__":
    # Train model
    model, best_loss, best_acc = main()
    
    # Optional: Load and evaluate
    # model = load_and_evaluate('best_st_gcn_model.pth')
