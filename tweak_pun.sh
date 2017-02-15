#!/usr/bin/env bash
KERAS_BACKEND=theano python app/tweak_params.py \
 --dataset=pun \
 --exp_name=$1 \
 --data_dir=data/pun/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=8000 \
 --max_sequence_len=20 \
 --num_epochs=20 \
 --tweak_max=10

