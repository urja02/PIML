# Physics-Informed Machine Learning (PIML) for Layered Material Analysis

This project implements machine learning models for predicting strain and stress in layered materials. It supports multiple model architectures including Graph Neural Networks (GNN), Feed-Forward Neural Networks (FNN), and Physics-informed Neural Networks (PNN).

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
  - [GNN Training](#gnn-training)
  - [FNN Training](#fnn-training)
  - [PNN Training](#pnn-training)
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
pip install -e .
```

## Project Structure

Key components of the codebase:
- `main.py`: Entry point for training and evaluation
- `train_GNN.py`: Graph Neural Network implementation
- `train_FNN.py`: Feed-Forward Neural Network implementation
- `train_PNN.py`: Physics-informed Neural Network implementation
- `data_preprocessing.py`: Data filtering and preprocessing utilities
- `dataset_generation.py`: Synthetic dataset generation
- `graphs_formation.py`: Graph construction for GNN
- `model.py`: Model architectures (GCN and GAT)

## Data Preparation

1. Download required datasets:
   - [Frame Large Dataset](https://drive.google.com/file/d/1jdqzxtWYaD6kauOPkwVkGCuL7iV14o_8/view?usp=drive_link)
   - [Section Dataset](https://drive.google.com/file/d/1A0mF4MpVF0N1zlPWxg8BFn4LPhnoTJnk/view?usp=drive_link)

2. Place the downloaded datasets in `PIML/training/data/`

3. Set up LayeredElastic dependency:
```bash
cd PIML/training/
git clone https://github.com/egemenokte/3DLayeredElastic.git LayeredElastic
```

## Training Models

### GNN Training
```bash
python main.py \
  --mode train \
  --model GNN \
  --frame_large_path training/data/frame_large.pkl \
  --section_path training/data/section.pkl \
  --lr 0.01 \
  --epochs 1000 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir logs/gnn_training
```

### FNN Training
```bash
python main.py \
  --mode train \
  --model FNN \
  --frame_large_path training/data/frame_large.pkl \
  --section_path training/data/section.pkl \
  --lr 0.001 \
  --epochs 500 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir logs/fnn_training
```

### PNN Training
```bash
python main.py \
  --mode train \
  --model PNN \
  --frame_large_path training/data/frame_large.pkl \
  --section_path training/data/section.pkl \
  --lr 0.001 \
  --epochs 500 \
  --optimizer Adam \
  --criterion L1loss \
  --log_dir logs/pnn_training
```

## Command Line Arguments

- `--mode`: Training or evaluation mode (`train` or `eval`)
- `--model`: Model architecture (`GNN`, `FNN`, or `PNN`)
- `--frame_large_path`: Path to frame_large dataset
- `--section_path`: Path to section dataset
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

## Dataset Access

Complete dataset collection is available at:
[PIML Dataset Collection](https://drive.google.com/drive/u/1/folders/1T2EYd7iKodO1UYzjE3eJXYsuqATpXhM-)

## Model Evaluation

For detailed model evaluation and analysis:
1. Use the provided Jupyter notebooks in the repository
2. Check the log directory specified during training for:
   - Training/validation loss curves
   - Model checkpoints
   - Performance metrics

## Citation

If you use this code in your research, please cite:
[Add citation information when available]
