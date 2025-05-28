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
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create model
    input_size = x_train.shape[1]
    model = FNNModel(input_size)
    
    # Setup training
    weight_dir = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x_train_tensor = x_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    x_val_tensor = x_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)

    batch_size = 1024
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
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
        for batch_x,batch_y in train_loader:

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
            # # Add L1 regularization
            # l1_loss = model.l1_regularization()
            # loss = loss + l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_tensor)
            loss_val = criterion(outputs_val, y_val_tensor)
            val_losses.append(loss_val.item())
        
        # Log progress
        if (epoch == 0) or (epoch + 1) % 100 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{args.epochs}], '
                f'Train Loss: {loss.item():.4f}, '
                f'Val Loss: {loss_val.item():.4f}'
            )
            # Save model checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(weight_dir, f"model_epoch_{epoch+1}.pt")
            )
    
