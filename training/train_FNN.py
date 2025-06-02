import os
import seaborn as sns
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import setup_logging
import logging
from typing import List, Tuple, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from PIML.model import FNNModel

def training_FNN(
    args: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """Train a feed-forward neural network model.
    
    Args:
        args: Command line arguments containing:
            - lr: Learning rate
            - epochs: Number of training epochs
            - optimizer: Optimizer type
            - criterion: Loss function type
            - log_dir: Directory for saving logs
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Create model
    input_size = x_train.shape[1]  # Get input size from data
    model = FNNModel(input_size)
    
    # Setup training
    weight_dir = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Convert to tensors and create dataloaders
    batch_size = 1024
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training setup
    if args.criterion == "L1loss":
        criterion = nn.L1Loss()
    elif args.criterion == "MSE":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion type: {args.criterion}")
        
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        # Training step
        model.train()
        epoch_train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation step
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_losses.append(loss.item())
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        # Log progress
        if (epoch == 0) or (epoch + 1) % 100 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{args.epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}'
            )
            # Save model checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(weight_dir, f"model_epoch_{epoch+1}.pt")
            )
    
