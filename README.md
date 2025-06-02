# Physics-Informed Machine Learning (PIML) for Layered Material Analysis

This project implements machine learning models for predicting strain and stress in layered materials. It supports multiple model architectures including Graph Neural Networks (GNN), Feed-Forward Neural Networks (FNN), and Physics-informed Neural Networks (PNN).

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Generation](#dataset-generation)
- [Training Models](#training-models)
  - [GNN Training](#gnn-training)
  - [FNN Training](#fnn-training)
  - [PNN Training](#pnn-training)
- [Model Evaluation](#model-evaluation)
- [Dataset Access](#dataset-access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/urja02/PIML.git
cd PIML
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:

```
PIML/
├── data/                  # Data storage directory
├── evaluation/            # Evaluation scripts
│   ├── eval_GNN.py       # GNN model evaluation
│   ├── eval_FNN.py       # FNN model evaluation
│   ├── eval_PNN.py       # PNN model evaluation
│   └── utils.py          # Evaluation utilities
├── training/             # Training implementations
│   ├── train_GNN.py      # GNN training implementation
│   ├── train_FNN.py      # FNN training implementation
│   ├── train_PNN.py      # PNN training implementation
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── dataset_generation.py  # Dataset generation scripts
│   ├── graphs_formation.py    # Graph construction for GNN
│   └── utils.py          # Training utilities
├── LayeredElastic/       # Layered elastic analysis components
├── plots/                # Directory for generated plots
├── main.py              # Main entry point
├── model.py             # Model architectures
├── requirements.txt     # Project dependencies
└── run scripts/         # Various execution scripts
    ├── run_dataset_generation.sh
    ├── run_train_*.sh   # Training scripts for each model
    └── run_eval_*.sh    # Evaluation scripts for each model
```

## Dataset Generation

To generate a new dataset, use the provided script:

```bash
./run_dataset_generation.sh
```

Or run directly with Python:

```bash
python main.py \
  --run_analysis \
  --data_path data \
  --mode train \
  --model GNN \
  --lr 0.01 \
  --epochs 1 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir training/log
```

## Training Models

### GNN Training
```bash
./run_train_gnn.sh
```

### FNN Training
```bash
./run_train_fnn.sh
```

### PNN Training
```bash
./run_train_pnn.sh
```

Each training script can be configured through command line arguments:
- `--mode`: Training mode (`train` or `eval`)
- `--model`: Model architecture (`GNN`, `FNN`, or `PNN`)
- `--data_path`: Data directory path
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--optimizer`: Optimization algorithm
- `--criterion`: Loss function
- `--log_dir`: Log directory path

## Model Evaluation

To evaluate trained models:

```bash
# For GNN evaluation
./run_eval_gnn.sh

# For FNN evaluation
./run_eval_fnn.sh

# For PNN evaluation
./run_eval_pnn.sh
```

The evaluation scripts provide:
- Model performance metrics
- Prediction accuracy analysis
- Visualization of results
- Comparison with ground truth

## Dataset Access

Pre-generated datasets are available at:
[PIML Dataset Collection](https://drive.google.com/drive/folders/1HLT3-ctCmgP86KtTfzyJd_QPH4wtlzWh?usp=sharing)

Download the datasets to the `data/` directory before running training or evaluation.

## Data Structure

The training data includes:
- Features (columns 3-13): Material properties and geometry
- Targets (columns 19-22): Strain components
- Data splits:
  - Training: First 135,520 samples
  - Validation: Samples 135,520 to 152,720
  - Testing: Samples 152,720 to 169,799

## Citation


