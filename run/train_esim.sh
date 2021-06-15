if [ "$#" -gt 0 ]
  then
      args="$*"
fi
hm=""
python3 ../fusion_run.py \
    --batch_size 16 \
    --proj_name train_esim  \
    --model esim \
    --embed_type 1  \
    --n_layer 6     \
    --d_embed 300   \
    --d_model 300  \
    --dropout 0.5  \
    --dropatt 0.15   \
    --optim adam    \
    --lr 0.0004    \
    --eta_min 0.00003 \
    --scheduler inv_sqrt \
    --warmup_step 0 \
    --epochs 10 \
    --cuda              \
    --model_name bert \
    --dataset_path $hm/remote_workspace/dataset/default \
    --examples_path $hm/remote_workspace/fusion_esim/data/bert_with_eot \
    --distill_dataset $hm/remote_workspace/fusion_esim/checkpoints/bert_eval/20210613-075809 \
    --bert_path $hm/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-12_H-768_A-12 \
    --distill_step \
    --load_post_trained_bert $hm/remote_workspace/fusion_esim/checkpoints/bert_post_train/20210608-221516/bert_2.pth.tar \
    --temperature 1 \
    $args
