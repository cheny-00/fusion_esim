if [ "$#" -gt 0 ]
  then
      args="$*"
fi
hm=""
python3 ../bert_eval.py \
    --batch_size 16 \
    --proj_name bert_eval  \
    --cuda              \
    --model_name bert \
    --dataset_path $hm/remote_workspace/dataset/default \
    --examples_path $hm/remote_workspace/fusion_esim/data/bert_with_eot \
    --bert_path $hm/remote_workspace/fusion_esim/data/pre_trained_ckpt/uncased_L-12_H-768_A-12 \
    --checkpoint_file $hm/remote_workspace/bert_post_train_and_fine_tuning/ESIM_like_3.pth.tar  \
    $args
