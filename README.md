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
- [Dataset Access](#dataset-access)
- [Model Evaluation](#model-evaluation)
- [Evaluation Notebooks](#evaluation-notebooks)

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
pip install -e .
```

## Project Structure

Key components of the codebase:
- `core/`: Core functionality shared between training and evaluation
  - `main.py`: Entry point for training, evaluation, and dataset generation
  - `LayeredElastic/`: Layered elastic analysis components
- `training/`: Training-specific components
  - `train_GNN.py`: Graph Neural Network implementation
  - `train_FNN.py`: Feed-Forward Neural Network implementation
  - `train_PNN.py`: Physics-informed Neural Network implementation
  - `data_preprocessing.py`: Data filtering and preprocessing utilities
  - `dataset_generation.py`: Synthetic dataset generation
  - `graphs_formation.py`: Graph construction for GNN
  - `model.py`: Model architectures (GCN and GAT)
- `data/`: Generated datasets and model inputs

## Dataset Access

Pre-generated datasets are available at:
[PIML Dataset Collection](https://drive.google.com/drive/folders/1HLT3-ctCmgP86KtTfzyJd_QPH4wtlzWh?usp=sharing)

Please download all the files to the data directory.

## Dataset Generation

To generate a new dataset for training or evaluation:

```bash
python PIML/core/main.py \
  --run_analysis \
  --data_path PIML/data \
  --mode train \
  --model GNN \
  --lr 0.01 \
  --epochs 1 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir PIML/training/log
```

The `--run_analysis` flag triggers the generation of:
1. Frame and section pickle files in the specified data directory:
   - `frame_large.pkl`: Contains material properties and responses
   - `section.pkl`: Contains layer geometries and configurations
2. Associated files needed for evaluation
3. Query points and analysis results

This process can take several hours depending on the number of samples and complexity of the analysis.

## Training Models

### GNN Training
```bash
python PIML/core/main.py \
  --mode train \
  --model GNN \
  --data_path PIML/data \
  --lr 0.01 \
  --epochs 1000 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir PIML/training/log
```

### FNN Training
```bash
python PIML/core/main.py \
  --mode train \
  --model FNN \
  --data_path PIML/data \
  --lr 0.001 \
  --epochs 500 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir PIML/training/log
```

### PNN Training
```bash
python PIML/core/main.py \
  --mode train \
  --model PNN \
  --data_path PIML/data \
  --lr 0.001 \
  --epochs 500 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir PIML/training/log
```

## Command Line Arguments

- `--run_analysis`: Flag to generate new dataset and analysis files
- `--mode`: Training or evaluation mode (`train` or `eval`)
- `--model`: Model architecture (`GNN`, `FNN`, or `PNN`)
- `--data_path`: Directory where frame_large.pkl and section.pkl will be stored/loaded
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--optimizer`: Optimization algorithm (`Adam` supported)
- `--criterion`: Loss function (`L1loss` or `MSE`)
- `--log_dir`: Directory for saving logs and checkpoints

## Data Structure

The training data is structured as follows:
- Features (columns 3-13): Material properties and geometry
- Targets (columns 19-22): Strain components
- Data splits:
  - Training: First 135,520 samples
  - Validation: Samples 135,520 to 152,720
  - Testing: Samples 152,720 to 169,799

## Model Evaluation

For detailed model evaluation and analysis:
1. Use the provided Jupyter notebooks in the repository
2. Check the log directory specified during training for:
   - Training/validation loss curves
   - Model checkpoints
   - Performance metrics

## Evaluation Notebooks

The project includes comprehensive evaluation notebooks for each model architecture. These notebooks provide detailed analysis, visualization, and performance metrics for the trained models.

### Available Evaluation Notebooks

1. **GAT Evaluation Notebook** (`evaluation/GAT_evaluation.ipynb`)
   - Evaluates Graph Attention Network (GAT) model performance
   - Includes model architecture description and visualization
   - Provides strain prediction analysis and heatmap generation

2. **FNN Evaluation Notebook** (`evaluation/FNN_evaluation.ipynb`)
   - Evaluates Feed-Forward Neural Network model performance
   - Features data preprocessing and model definition sections
   - Includes comprehensive performance metrics and visualizations

3. **PNN Evaluation Notebook** (`evaluation/PNN_evaluation.ipynb`)
   - Evaluates Physics-Informed Neural Network model performance
   - Contains detailed model architecture documentation
   - Provides strain component analysis and comparison plots

### Required Data Files

Before running the evaluation notebooks, ensure you have the following files in your data directory:

- `batched_graph_test.pkl`: Test dataset in graph format
- `frame_large.pkl`: Full dataset with material properties and responses
- `section.pkl`: Layered structure information
- `ZS.pkl`: Z-coordinates (depth points) for each section
- `xs.pkl`: X-coordinates (radial distance points)
- Model files: `gnn_model.pth`, `fnn_model.pth`, `pnn_model.pth`

### Colab Links

Direct links to run the notebooks in Google Colab:

- [GAT Evaluation Notebook](https://colab.research.google.com/github/urja02/PIML/blob/main/evaluation/GAT_evaluation.ipynb)
- [FNN Evaluation Notebook](https://drive.google.com/file/d/1QXBGDEgXzHIziuW4fRaxW_6mYaMjno6p/view?usp=drive_link)
- [PNN Evaluation Notebook](https://colab.research.google.com/drive/1GD9zqO5qQtStgfwHyRx4cbITLApQMmcv?usp=drive_link)

**Note:** Make sure to upload the required data files to your Google Drive before running the notebooks.

## Citation

If you use this code in your research, please cite:
[Add citation information when available]
