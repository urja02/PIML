python main.py \
    --mode eval \
    --model FNN \
    --data_path data \
    --lr 0.01 \
    --epochs 10 \
    --optimizer Adam \
    --criterion MSE \
    --log_dir training/log \
    --model_path data/fnn_model.pth 