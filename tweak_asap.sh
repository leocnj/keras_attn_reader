#!/usr/bin/env bash
# gpu item_id batch_size (16, 32, 48)
KERAS_BACKEND=theano THEANO_FLAGS=device=${1} python app/tweak_params.py \
 --dataset=asap \
 --exp_name=lstm \
 --data_dir=data/asap/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=4500 \
 --max_sequence_len=75 \
 --num_epochs=30 \
 --batch_size=${3} \
 --tweak_max=100 \
 --tweak_id=${2}
