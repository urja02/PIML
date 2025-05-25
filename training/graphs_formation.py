import torch
import numpy as np
from typing import List, Tuple
from torch_geometric.data import Data, Batch

def create_graph(inputs: np.ndarray, outputs: np.ndarray, edge_index: np.ndarray) -> Data:
    """Create a single graph from input data, outputs and edge indices.
    
    Args:
        inputs: Node features array
        outputs: Target values array
        edge_index: Edge indices array
        
    Returns:
        Data: PyTorch Geometric graph data object
    """
    # Convert inputs to torch tensor and ensure float type
    inputs = torch.from_numpy(inputs).float()
    
    # Convert edge index to torch tensor and transpose
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create graph using PyTorch Geometric data structure
    graph = Data(
        x=inputs,  # Node features
        edge_index=edge_index,  # Edge indices
        y=torch.from_numpy(outputs).float()  # Target values
    )
    
    return graph

def create_batched_graph(data: List[np.ndarray], 
                        outputs: List[np.ndarray], 
                        edge_indices: List[np.ndarray]) -> Batch:
    """Create a batched graph from lists of data.
    
    Args:
        data: List of input arrays
        outputs: List of output arrays
        edge_indices: List of edge index arrays
        
    Returns:
        Batch: Batched PyTorch Geometric graphs
    """
    graphs = []
    
    for inputs, targets, edges in zip(data, outputs, edge_indices):
        graph = create_graph(inputs, targets, edges)
        graphs.append(graph)
        
    return Batch.from_data_list(graphs)

def train_graph(TRAIN: List[np.ndarray], 
                TRAIN_out: List[np.ndarray], 
                MAT_edge: List[np.ndarray], 
                MAT_dist: List[np.ndarray]) -> Batch:
    """Create batched graph for training data.
    
    Args:
        TRAIN: List of training input arrays
        TRAIN_out: List of training output arrays
        MAT_edge: List of edge index arrays
        MAT_dist: List of edge distance arrays (unused)
        
    Returns:
        Batch: Batched training graphs
    """
    return create_batched_graph(TRAIN, TRAIN_out, MAT_edge)

def val_graph(VAL: List[np.ndarray], 
              VAL_out: List[np.ndarray], 
              MAT_edge_val: List[np.ndarray], 
              MAT_dist_val: List[np.ndarray]) -> Batch:
    """Create batched graph for validation data.
    
    Args:
        VAL: List of validation input arrays
        VAL_out: List of validation output arrays
        MAT_edge_val: List of edge index arrays
        MAT_dist_val: List of edge distance arrays (unused)
        
    Returns:
        Batch: Batched validation graphs
    """
    return create_batched_graph(VAL, VAL_out, MAT_edge_val)

def test_graph(TEST: List[np.ndarray], 
               TEST_out: List[np.ndarray], 
               MAT_edge_test: List[np.ndarray], 
               MAT_dist_test: List[np.ndarray]) -> Batch:
    """Create batched graph for test data.
    
    Args:
        TEST: List of test input arrays
        TEST_out: List of test output arrays
        MAT_edge_test: List of edge index arrays
        MAT_dist_test: List of edge distance arrays (unused)
        
    Returns:
        Batch: Batched test graphs
    """
    return create_batched_graph(TEST, TEST_out, MAT_edge_test)