python3 ../fusion_run.py \
    --batch_size 16 \
    --model fusion_esim \
    --embed_type 1  \
    --n_layer 6     \
    --d_embed 300   \
    --d_model 300  \
    --dropout 0.5  \
    --dropatt 0.15   \
    --optim adam    \
    --lr 0.0004    \
    --eta_min 0.00003 \
    --scheduler cosine_warm_up \
    --warmup_step 8100 \
    --epochs 64 \
    --cuda              \
    --model_name bert \
    --dataset_path /remote_workspace/dataset/default \
    --examples_path /remote_workspace/rs_trans/data/examples \
    --fp16 \
    --opt_level O2