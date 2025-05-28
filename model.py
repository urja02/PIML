import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List

class GAT(nn.Module):
    """Graph Attention Network (GAT) implementation.
    
    This model uses multiple Graph Attention layers followed by linear layers
    to process graph-structured data.
    
    Attributes:
        convs: List of Graph Attention Convolution layers
        batch_norms: List of batch normalization layers
        linear1: First linear layer for final processing
        linear2: Second linear layer for output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        """Initialize the GAT model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim1: Dimension of first hidden layer
            hidden_dim: Dimension of subsequent hidden layers
            output_dim: Dimension of output
            num_layers: Number of GAT layers
        """
        super(GAT, self).__init__()
        torch.manual_seed(1234)  # For reproducibility
        
        # Helper functions to determine layer dimensions
        def get_in_channels(idx: int) -> int:
            if idx > 1:
                return hidden_dim
            elif idx == 1:
                return hidden_dim1
            else:
                return input_dim
                
        def get_out_channels(idx: int) -> int:
            if idx == 0:
                return hidden_dim1
            elif idx > 0 and idx <= num_layers - 1:
                return hidden_dim
            else:
                return output_dim
        
        # Create GAT layers with corresponding batch norm layers
        self.convs = nn.ModuleList([
            GATConv(
                in_channels=get_in_channels(i),
                out_channels=get_out_channels(i)
            ) for i in range(num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(get_out_channels(i))
            for i in range(num_layers)
        ])
        
        # Final linear layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input node features tensor
            edge_index: Graph connectivity tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Apply GAT layers with batch norm and ReLU
        for gat, batch_norm in zip(self.convs, self.batch_norms):
            x = gat(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
        
        # Apply final MLP layers
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear1(x)  # Reusing linear1 as in original
        x = F.relu(x)
        
        return self.linear2(x)

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

class NeuralNetwork(nn.Module):
    """Simple feed-forward neural network implementation for PNN."""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        input_size = 5
        hidden_size = 90
        output_size = 3
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
