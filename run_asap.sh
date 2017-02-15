#!/usr/bin/env bash
KERAS_BACKEND=theano python app/train_cv.py \
 --exp_name=att\
 --data_dir=../dl_response_rater/data/asap2/ \
 --embedding_file_path=embd/glove.6B.300d.txt \
 --embedding_dim=300 \
 --nb_words=4500\
 --max_sequence_len=75\
 --lstm_hs=64 \
 --model_name=lstm-other

