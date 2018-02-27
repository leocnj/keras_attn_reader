#!/usr/bin/env bash
# exp_name ET_yes batch_size
#
KERAS_BACKEND=tensorflow python app/runexp_cv.py \
 --dataset=asap \
 --exp_name=${1} \
 --data_dir=data/asap/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --embedding_trainable=${2} \
 --nb_words=4500 \
 --max_sequence_len=75 \
 --num_epochs=30 \
 --batch_size=${3}

