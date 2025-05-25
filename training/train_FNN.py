import os
import seaborn as sns
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging
import logging
from typing import List, Tuple, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class FNNModel(nn.Module):
    def __init__(self, input_size: int, l1_lambda: float = 1e-5):
        super(FNNModel, self).__init__()
        self.l1_lambda = l1_lambda
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 3)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x

    def l1_regularization(self) -> torch.Tensor:
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return self.l1_lambda * l1_norm

def training_FNN(
    args: Any,
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    input_val: List[np.ndarray],
    target_val: List[np.ndarray],
    test: List[np.ndarray],
    target_test: List[np.ndarray]
) -> None:
    """Train a feed-forward neural network model.
    
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
        target_val: List containing validation target array
        test: List containing test input array
        target_test: List containing test target array
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Prepare data - data is already split in main.py
    x_train = inputs[0]  # First element of the list
    y_train = targets[0]
    x_val = input_val[0]
    y_val = target_val[0]
    x_test = test[0]
    y_test = target_test[0]
    
    # Scale the data
    scaler = StandardScaler()
    scalery = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    y_train = scalery.fit_transform(y_train)
    y_val = scalery.transform(y_val)
    y_test = scalery.transform(y_test)
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
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
    
