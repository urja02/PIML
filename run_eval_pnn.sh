python main.py \
    --mode eval \
    --model PNN \
    --data_path data \
    --lr 0.001 \
    --epochs 15000 \
    --optimizer Adam \
    --criterion L1loss \
    --log_dir training/log \
    --model_path training/log/2/checkpoints/model_epoch_6999.pt 