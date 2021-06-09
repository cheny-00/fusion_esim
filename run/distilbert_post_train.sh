if [ "$#" -gt 0 ]
  then
      args="$*"
fi

python3 ../distilbert_post.py  \
  --batch_size 8 \
  --cuda \
  --bert_path ../data/pre_trained_ckpt/distilbert-base-uncased  \
  --proj_name distilbert_post_train \
    $args
