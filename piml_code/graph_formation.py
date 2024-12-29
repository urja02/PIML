import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader, Batch

def train_graph(TRAIN, TRAIN_out, MAT_edge, MAT_dist):
    graphs = []

    for i in range(len(TRAIN)):
        inputs = TRAIN[i]  # Extract inputs for the i-th sample
        outputs = TRAIN_out[i]  # Extract outputs for the i-th sample

        # Assuming inputs contains the coordinates (x, z) for nodes
        # Assuming outputs contains the target values
        inputs=torch.from_numpy(inputs)
        inputs = inputs.float()

        targets=torch.from_numpy(outputs)
        targets= targets.float()
        # Example: Calculate edge indices for the i-th input
        edge_index = MAT_edge[i]
        edge_index=torch.tensor(edge_index)
        edge_index = torch.t(edge_index)
        edge_distance = MAT_dist[i]

        # Create a graph using PyTorch Geometric data structure
        # Constructing Data object
        graph = Data(x=torch.tensor(inputs, dtype=torch.float),  # Node features
                        edge_index=torch.tensor(edge_index, dtype=torch.long),  # Edge indices
                        y=torch.tensor(outputs, dtype=torch.float))  # Target values

        # Append the graph to the list
        graphs.append(graph)

    batched_graph = Batch.from_data_list(graphs)
    return batched_graph

def val_graph(VAL, VAL_out, MAT_edge_val, MAT_dist_val):
    graphs_val = []

    for i in range(len(VAL)):
        inputs = VAL[i]  # Extract inputs for the i-th sample
        outputs = VAL_out[i]  # Extract outputs for the i-th sample

        # Assuming inputs contains the coordinates (x, z) for nodes
        # Assuming outputs contains the target values
        inputs=torch.from_numpy(inputs)
        inputs = inputs.float()

        targets=torch.from_numpy(outputs)
        targets= targets.float()
        # Example: Calculate edge indices for the i-th input
        edge_index = MAT_edge_val[i]
        edge_index=torch.tensor(edge_index)
        edge_index = torch.t(edge_index)
        edge_distance = MAT_dist_val[i]

        # Create a graph using PyTorch Geometric data structure
        # Constructing Data object
        graph = Data(x=torch.tensor(inputs, dtype=torch.float),  # Node features
                        edge_index=torch.tensor(edge_index, dtype=torch.long),  # Edge indices
                        y=torch.tensor(outputs, dtype=torch.float))  # Target values

        # Append the graph to the list
        graphs_val.append(graph)
    batched_graph_val = Batch.from_data_list(graphs_val)
    return batched_graph_val

def test_graph(TEST, TEST_out, MAT_edge_test, MAT_dist_test):
    graphs_test = []
    for i in range(len(TEST)):
        inputs = TEST[i]  # Extract inputs for the i-th sample
        outputs = TEST_out[i]  # Extract outputs for the i-th sample

        # Assuming inputs contains the coordinates (x, z) for nodes
        # Assuming outputs contains the target values
        inputs=torch.from_numpy(inputs)
        inputs = inputs.float()

        targets=torch.from_numpy(outputs)
        targets= targets.float()
        # Example: Calculate edge indices for the i-th input
        edge_index = MAT_edge_test[i]
        edge_index=torch.tensor(edge_index)
        edge_index = torch.t(edge_index)
        edge_distance = MAT_dist_test[i]

        # Create a graph using PyTorch Geometric data structure
        # Constructing Data object
        graph = Data(x=torch.tensor(inputs, dtype=torch.float),  # Node features
                        edge_index=torch.tensor(edge_index, dtype=torch.long),  # Edge indices
                        y=torch.tensor(outputs, dtype=torch.float))  # Target values

        # Append the graph to the list
        graphs_test.append(graph)
    batched_graph_test = Batch.from_data_list(graphs_test)
    return batched_graph_test