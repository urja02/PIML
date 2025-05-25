import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from dataset_generation import generatesection, plot_sample_query_points,generate_query_points,analysis
from data_preprocessing import frame_filtering,  filtering_ZS, train_val_test_generate, formation_of_matrices,remove_strain_z
from graphs_formation import train_graph, val_graph, test_graph
from train_GNN import training_loop
from model import GAT
import logging
from train_PNN import training_nn
from train_FNN import training_FNN
import numpy as np

# Constants
MATERIAL_CONFIG = {
    'N_MATERIALS': 3,
    'MATERIAL_TYPES': ['AC', 'B', 'SG'],  # AC, base, subbase, subgrade
    'SUBLAYER_MAX': [2, 2, 1],  # each material's max sublayers, excluding subgrade
    'THICKNESS_RANGE': [[2, 16], [4, 20]],  # thickness range in inches
    'MODULUS_RANGE': [[500, 2000], [50, 300], [5, 50]],  # modulus range in ksi
    'THICKNESS_INCREMENT': [1, 2, 4],
    'MODULUS_INCREMENT': [50, 20, 20, 5],  # increment in modulus sampling
    'NU_RANGE': [[0.3, 0.4], [0.2, 0.499], [0.2, 0.499]]  # poissons ratio
}

SAMPLING_CONFIG = {
    'N_POINTS': 1000,  # Number of points
    'Z_POINTS': 14,  # points to generate along z
    'X_POINTS': 10,  # points to generate along x
    'A_RANGE': [4, 4],  # contact radius (in)
    'A_POINTS': 1,  # how many contact radii to analyze
    'FACTOR': 0.4,
    'FILTER': 2,
    'SPLIT_IDX': 800,
    'TEST_IDX': 900,
    'SEED': 42
}

MODEL_CONFIG = {
    'INPUT_DIM': 5,
    'HIDDEN_DIM1': 128,
    'HIDDEN_DIM': 90,
    'OUTPUT_DIM': 3,
    'NUM_LAYERS': 10
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the training/evaluation pipeline.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="PIML Training/Evaluation Pipeline")
    
    # Mode arguments
    parser.add_argument("--run_analysis", action="store_true", default=False, 
                       help="Set this flag to run analysis")
    parser.add_argument("--mode", type=str, choices=["train", "eval"],
                       help="Training or evaluation mode")
    parser.add_argument("--model", type=str, choices=["GNN","FNN","PNN"],
                       help="Use neural network model")
    
    # Data arguments
    parser.add_argument("--frame_large_path", type=str, required=True,
                       help="Path to FrameLarge pickle file")
    parser.add_argument("--section_path", type=str, required=True,
                       help="Path to Section pickle file")
    
    # Model arguments
    parser.add_argument("--lr", type=float, required=True,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True,
                       help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="Adam",
                       help="Optimizer type")
    parser.add_argument("--criterion", type=str, required=True,
                       help="Loss function type")
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Directory for saving logs")
    
    return parser.parse_args()

def choose_optimizer(
    model: nn.Module,
    optimizer_type: str,
    lr: float
) -> torch.optim.Optimizer:
    """Create optimizer for model training.
    
    Args:
        model: Neural network model
        optimizer_type: Type of optimizer
        lr: Learning rate
        
    Returns:
        torch.optim.Optimizer: Created optimizer
    """
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_criterion(criterion_type: str) -> nn.Module:
    """Get loss criterion based on specified type.
    
    Args:
        criterion_type: Type of loss function
        
    Returns:
        nn.Module: Loss criterion
    """
    if criterion_type == "L1loss":
        return nn.L1Loss()
    elif criterion_type == "MSE":
        return nn.MSELoss()
    raise ValueError(f"Unsupported criterion type: {criterion_type}")

def load_or_generate_data(
    args: argparse.Namespace,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing data or generate new data based on configuration.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: FrameLarge and Section data
    """
    if args.run_analysis:
        Section, Frame = generatesection(
            config['N_POINTS'], config['N_MATERIALS'],
            config['MATERIAL_TYPES'], config['SUBLAYER_MAX'],
            config['THICKNESS_RANGE'], config['MODULUS_RANGE'],
            config['Z_POINTS'], config['X_POINTS'],
            config['THICKNESS_INCREMENT'], config['MODULUS_INCREMENT'],
            config['NU_RANGE'], config['A_RANGE'],
            config['A_POINTS'], seed=config['SEED']
        )
        
        FrameLarge, ZS, xs, E, NU, final_dict_ztoE, H, final_dict_ztoH, final_dict_ztonu = generate_query_points(
            Section, config['N_POINTS'], config['X_POINTS'],
            config['Z_POINTS'], config['FACTOR'],
            config['A_RANGE'], Frame
        )
        
        analysis(args.frame_large_path, args.section_path, FrameLarge, Section)
        
        with open(args.frame_large_path, 'rb') as fp:
            FrameLarge = pickle.load(fp)
        with open(args.section_path, 'rb') as fp:
            Section = pickle.load(fp)
    else:
        with open(args.frame_large_path, 'rb') as fp:
            FrameLarge = pickle.load(fp)
        with open(args.section_path, 'rb') as fp:
            Section = pickle.load(fp)
            
    return FrameLarge, Section

def main(args: argparse.Namespace):
    """Main training/evaluation pipeline.
    
    Args:
        args: Command line arguments
    """
    # Load or generate data
    FrameLarge, Section = load_or_generate_data(args, {**MATERIAL_CONFIG, **SAMPLING_CONFIG})
    
    # Generate temporary data for visualization
    Section_temp, Frame = generatesection(
        SAMPLING_CONFIG['N_POINTS'], MATERIAL_CONFIG['N_MATERIALS'],
        MATERIAL_CONFIG['MATERIAL_TYPES'], MATERIAL_CONFIG['SUBLAYER_MAX'],
        MATERIAL_CONFIG['THICKNESS_RANGE'], MATERIAL_CONFIG['MODULUS_RANGE'],
        SAMPLING_CONFIG['Z_POINTS'], SAMPLING_CONFIG['X_POINTS'],
        MATERIAL_CONFIG['THICKNESS_INCREMENT'], MATERIAL_CONFIG['MODULUS_INCREMENT'],
        MATERIAL_CONFIG['NU_RANGE'], SAMPLING_CONFIG['A_RANGE'],
        SAMPLING_CONFIG['A_POINTS'], seed=SAMPLING_CONFIG['SEED']
    )

    # Plot sample query points
    plot_sample_query_points(Section_temp, SAMPLING_CONFIG['X_POINTS'],
                           SAMPLING_CONFIG['Z_POINTS'], SAMPLING_CONFIG['FACTOR'])

    # Generate query points
    FrameLarge_temp, ZS, xs, E, NU, final_dict_ztoE, H, final_dict_ztoH, final_dict_ztonu = generate_query_points(
        Section, SAMPLING_CONFIG['N_POINTS'], SAMPLING_CONFIG['X_POINTS'],
        SAMPLING_CONFIG['Z_POINTS'], SAMPLING_CONFIG['FACTOR'],
        SAMPLING_CONFIG['A_RANGE'], Frame
    )

    if args.model == "PNN":
        # Neural network training path
        TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out, ZS_train, ZS_val, ZS_test, mins_train, maxs_train = train_val_test_generate(
            FrameLarge, final_dict_ztoE, ZS, xs,
            SAMPLING_CONFIG['SPLIT_IDX'], SAMPLING_CONFIG['TEST_IDX'],
            SAMPLING_CONFIG['N_POINTS'], final_dict_ztoH, final_dict_ztonu
        )
        training_nn(args, TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out)

    elif args.model == "FNN":
        # Feed-forward neural network training path
        # Split data using specific indices
        ZS, DF = remove_strain_z(FrameLarge)
        x_train = DF.iloc[:135520, 3:13].values
        y_train = DF.iloc[:135520, 19:22].values
        x_val = DF.iloc[135520:152720, 3:13].values
        y_val = DF.iloc[135520:152720, 19:22].values
        x_test = DF.iloc[152720:169799, 3:13].values
        y_test = DF.iloc[152720:169799, 19:22].values
        
        # Convert to list format as expected by training_FNN
        TRAIN = [x_train]
        TRAIN_out = [y_train]
        VAL = [x_val]
        VAL_out = [y_val]
        TEST = [x_test]
        TEST_out = [y_test]
        
        training_FNN(args, TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out)

    elif args.model == "GNN":
        # Strain prediction path
        train_df = FrameLarge.loc[FrameLarge['Structure'] <= SAMPLING_CONFIG['SPLIT_IDX']]
        train_df = train_df[['Strain_Z', 'Strain_R', 'Strain_T']]

        ZS, DF = remove_strain_z(FrameLarge)

        # Generate train/val/test splits for GNN
        TRAIN, TRAIN_out, VAL, VAL_out, TEST, TEST_out, ZS_train, ZS_val, ZS_test, mins_train, maxs_train = train_val_test_generate(
            DF, final_dict_ztoE, ZS, xs,
            SAMPLING_CONFIG['SPLIT_IDX'], SAMPLING_CONFIG['TEST_IDX'],
            SAMPLING_CONFIG['N_POINTS'], final_dict_ztoH, final_dict_ztonu
        )
        
        # Generate matrices for graph formation
        MAT_edge, MAT_dist, MAT_edge_val, MAT_dist_val, MAT_edge_test, MAT_dist_test = formation_of_matrices(
            TRAIN, VAL, TEST, ZS_train, xs, ZS_val, ZS_test
        )

        # Create and setup model
        model = GAT(
            MODEL_CONFIG['INPUT_DIM'],
            MODEL_CONFIG['HIDDEN_DIM1'],
            MODEL_CONFIG['HIDDEN_DIM'],
            MODEL_CONFIG['OUTPUT_DIM'],
            MODEL_CONFIG['NUM_LAYERS']
        )

        # Create graph structures
        batched_graph = train_graph(TRAIN, TRAIN_out, MAT_edge, MAT_dist)
        batched_graph_val = val_graph(VAL, VAL_out, MAT_edge_val, MAT_dist_val)
        batched_graph_test = test_graph(TEST, TEST_out, MAT_edge_test, MAT_dist_test)

        # Setup training
        optimizer = choose_optimizer(model, args.optimizer, args.lr)
        crit = get_criterion(args.criterion)

        # Train model if in training mode
        if args.mode == "train":
            training_loop(
                args, model, optimizer,
                batched_graph, batched_graph_val, batched_graph_test,
                args.lr, args.epochs, crit,
                Section, ZS, xs, weighted_loss=None
            )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

