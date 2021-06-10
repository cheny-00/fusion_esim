if [ "$#" -gt 0 ]
  then
      args="$*"
fi

python3 ../post_train.py  \
  --batch_size 8 \
  --cuda \
    $args
