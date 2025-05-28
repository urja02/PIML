import os
import seaborn as sns
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import setup_logging
import logging
from typing import List, Tuple, Any, Optional
from pathlib import Path
from PIML.model import NeuralNetwork

def denormalize_min_max(
    normalized_data: np.ndarray,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """Denormalize data using min-max scaling.
    
    Args:
        normalized_data: Normalized input data
        min_val: Minimum value used in normalization
        max_val: Maximum value used in normalization
        
    Returns:
        np.ndarray: Denormalized data
    """
    print("denormalizing")
    print((normalized_data * (max_val - min_val)) + min_val)
    
    return (normalized_data * (max_val - min_val)) + min_val

def training_nn(
    args: Any,
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    input_val: List[np.ndarray],
    target_val: List[np.ndarray]
) -> None:
    """Train a neural network model.
    
    Args:
        args: Command line arguments containing:
            - lr: Learning rate
            - epochs: Number of training epochs
            - optimizer: Optimizer type
            - criterion: Loss function type
            - log_dir: Directory for saving logs
        inputs: List containing training input array
        targets: List containing training target array
        input_val: List containing validation input array
    """
    torch.manual_seed(42)  # For reproducibility
    

    # Create model
    model = NeuralNetwork()
    
    # Setup training
    weight_dir = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare data
    inputs_tensor = torch.tensor(np.vstack(inputs), dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(np.vstack(targets), dtype=torch.float32).to(device)
    input_val_tensor = torch.tensor(np.vstack(input_val), dtype=torch.float32).to(device)
    target_val_tensor = torch.tensor(np.vstack(target_val), dtype=torch.float32).to(device)
    
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
        outputs = model(inputs_tensor)
        loss = criterion(outputs, targets_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.mean().item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            outputs_val = model(input_val_tensor)
            loss_val = criterion(outputs_val, target_val_tensor)
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
                os.path.join(weight_dir, f"model_epoch_{epoch}.pt")
            )

    
        

