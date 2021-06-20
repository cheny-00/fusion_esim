if [ "$#" -gt 0 ]
  then
      args="$*"
fi
hm=""
python3 ../esim_run.py \
    --batch_size 32 \
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
    --scheduler dev_perf \
    --warmup_step 0 \
    --epochs 64 \
    --cuda              \
    --model_name bert \
    --dataset_path $hm/remote_workspace/dataset/default \
    --examples_path $hm/remote_workspace/fusion_esim/data/bert_with_eot \
    --distill_dataset /remote_workspace/fusion_esim/checkpoints/create_distillation_dataset/20210619-211256 \
    --bert_path $hm/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-12_H-768_A-12 \
    --distill_step \
    --temperature 1 \
    --eval_interval 3 \
    $args
   #--load_post_trained_bert $hm/remote_workspace/fusion_esim/checkpoints/bert_post_train/20210608-221516/bert_2.pth.tar \
