import time
import torch
import torch.nn as nn
import logging
import seaborn as sns
from typing import Optional, Dict, Any, List
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Batch
from pathlib import Path
from .utils import setup_logging
from PIML.model import GAT

def compute_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor
) -> Dict[str, float]:
    """Compute MSE metrics for each output dimension.
    
    Args:
        targets: Ground truth values
        predictions: Model predictions
        
    Returns:
        Dict[str, float]: Dictionary containing MSE for each dimension
    """
    targets_np = targets.cpu().detach().numpy()
    preds_np = predictions.cpu().detach().numpy()
    
    return {
        'mse_z': mean_squared_error(targets_np[:, 0], preds_np[:, 0]),
        'mse_r': mean_squared_error(targets_np[:, 1], preds_np[:, 1]),
        'mse_t': mean_squared_error(targets_np[:, 2], preds_np[:, 2])
    }

def train_step(
    model: nn.Module,
    data: Batch,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> Dict[str, float]:
    """Perform one training step.
    
    Args:
        model: Neural network model
        data: Batch of training data
        optimizer: Model optimizer
        criterion: Loss criterion
        
    Returns:
        Dict[str, float]: Dictionary containing loss and metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    metrics = compute_metrics(data.y, out)
    metrics['loss'] = loss.item()
    
    return metrics

def validate(
    model: nn.Module,
    data: Batch,
    criterion: nn.Module
) -> Dict[str, float]:
    """Perform validation.
    
    Args:
        model: Neural network model
        data: Batch of validation data
        criterion: Loss criterion
        
    Returns:
        Dict[str, float]: Dictionary containing validation loss and metrics
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_loss = criterion(out, data.y)
        
        metrics = compute_metrics(data.y, out)
        metrics['loss'] = val_loss.item()
        
        return metrics

def training_loop(
    args: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Batch,
    val_data: Batch,
    test_data: Batch,
    lr: float,
    epochs: int,
    criterion: nn.Module,
    section_data: Any,
    zs_data: List[float],
    xs_data: List[float],
    weighted_loss: Optional[bool] = None
) -> None:
    """Main training loop.
    
    Args:
        args: Command line arguments
        model: Neural network model
        optimizer: Model optimizer
        train_data: Training data batch
        val_data: Validation data batch
        test_data: Test data batch
        lr: Learning rate
        epochs: Number of epochs
        criterion: Loss criterion
        section_data: Section data
        zs_data: Z-coordinate data
        xs_data: X-coordinate data
        weighted_loss: Whether to use weighted loss
    """
    # Setup logging and move model to device
    weight_dir = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    # Training metrics history
    train_losses = []
    val_losses = []
    train_metrics = {
        'error_z': [],
        'error_r': [],
        'error_t': []
    }
    val_metrics = {
        'error_z': [],
        'error_r': [],
        'error_t': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training step
        train_results = train_step(model, train_data, optimizer, criterion)
        train_losses.append(train_results['loss'])
        train_metrics['error_z'].append(train_results['mse_z'])
        train_metrics['error_r'].append(train_results['mse_r'])
        train_metrics['error_t'].append(train_results['mse_t'])
        
        # Validation step
        val_results = validate(model, val_data, criterion)
        val_losses.append(val_results['loss'])
        val_metrics['error_z'].append(val_results['mse_z'])
        val_metrics['error_r'].append(val_results['mse_r'])
        val_metrics['error_t'].append(val_results['mse_t'])
        
        # Log progress
        if (epoch + 1) % 100 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{epochs}], '
                f'Train Loss: {train_results["loss"]:.4f}, '
                f'Val Loss: {val_results["loss"]:.4f}'
            )
            
            # Save model checkpoint
            checkpoint_path = Path(weight_dir) / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)




