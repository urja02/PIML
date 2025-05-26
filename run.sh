python 'PIML/core/main.py' \
    --mode train \
    --model GNN \
    --data_path PIML/data \
    --lr 0.01 \
    --epochs 1 \
    --optimizer Adam \
    --criterion L1loss \
    --log_dir PIML/training/log 