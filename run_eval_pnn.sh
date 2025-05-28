python main.py \
    --mode eval \
    --model PNN \
    --data_path data \
    --lr 0.01 \
    --epochs 10 \
    --optimizer Adam \
    --criterion L1loss \
    --log_dir training/log \
    --model_path data/pnn_model.pth 