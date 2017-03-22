#!/usr/bin/env bash
KERAS_BACKEND=theano THEANO_FLAGS=device=${2} python app/runexp_cv.py \
 --dataset=asap \
 --exp_name=${1} \
 --data_dir=data/asap/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --embedding_trainable=${3} \
 --nb_words=4500 \
 --max_sequence_len=75 \
 --num_epochs=30 \
 --batch_size=${4}

