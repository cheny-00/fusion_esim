if [ "$#" -gt 0 ]
  then
      args="$*"
fi
hm=""
python3 ../fusion_run.py \
    --batch_size 32 \
    --proj_name distilbert_fine_tuning  \
    --model fusion_esim \
    --embed_type 1  \
    --n_layer 6     \
    --d_embed 300   \
    --d_model 300  \
    --dropout 0.5  \
    --dropatt 0.15   \
    --optim adam    \
    --lr 0.004    \
    --eta_min 0.00003 \
    --scheduler inv_sqrt \
    --warmup_step 8100 \
    --epochs 5 \
    --cuda              \
    --model_name distilbert \
    --dataset_path $hm/remote_workspace/dataset/default \
    --examples_path $hm/remote_workspace/fusion_esim/data/bert_with_eot \
    --bert_path $hm/remote_workspace/fusion_esim/data/pre_trained_ckpt/distilbert-base-uncased \
    --fine_tuning \
    --load_post_trained_bert $hm/remote_workspace/fusion_esim/checkpoints/distilbert_post_train/20210610-002430/bert_2.pth.tar \
    $args
