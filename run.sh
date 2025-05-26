python 'core/main.py' \
    --mode train \
    --model GNN \
    --data_path data \
    --lr 0.01 \
    --epochs 1 \
    --optimizer Adam \
    --criterion L1loss \
    --log_dir training/log 