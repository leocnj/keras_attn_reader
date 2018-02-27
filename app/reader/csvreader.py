from __future__ import division
from __future__ import print_function

import re
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_input_csv(train_csv, test_csv, nb_words, maxlen):
    """
    Method to read csv file pairs

    :returns
    X_train (, maxlen)
    Y_train (, 2)
    X_test
    Y_Test
    word_index: will be used later for initing embedding matrix

    """
    train_df = pd.read_csv(train_csv)
    train_df = train_df.sample(frac=1).reset_index(drop=True) # shuffle train
    test_df = pd.read_csv(test_csv)
    print(train_df.head())

    train_X = train_df.text.values.tolist()
    test_X = test_df.text.values.tolist()

    train_y = train_df.label.values
    test_y = test_df.label.values
    nb_classes = len(np.unique(train_y))
    train_Y = np_utils.to_categorical(train_y, nb_classes)
    test_Y = np_utils.to_categorical(test_y, nb_classes)

    # tokenrize should be applied on train+test jointly
    n_ta = len(train_X)
    n_ts = len(test_X)
    print('Instances: train {} vs. test {}'.format(n_ta, n_ts))

    # textraw = [line.encode('utf-8') for line in train_X+test_X]  # keras needs str
    textraw = [str(line) for line in train_X+test_X]  # python3
    token = Tokenizer(nb_words=nb_words)
    token.fit_on_texts(textraw)

    word_index = token.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    textseq = token.texts_to_sequences(textraw)
    lens = [len(line) for line in textseq]
    print('sentence lens: {}'.format([np.min(lens), np.max(lens), np.percentile(lens, 90)]))

    train_X = pad_sequences(textseq[0:n_ta], maxlen,  padding='post', truncating='post')
    test_X = pad_sequences(textseq[n_ta:], maxlen,  padding='post', truncating='post')

    train_len = np.asarray(lens[0:n_ta], dtype='float32')
    test_len = np.asarray(lens[n_ta:], dtype='float32')

    return train_X, train_Y, test_X, test_Y, word_index, nb_classes, train_len, test_len


def test_pun2():
    """
    test code
    :return:
    """
    dir = "../../data/pun/"
    ta_csv = dir + "train_cv1.csv"
    ts_csv = dir + "validation.csv"
    tps = read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)

    print(tps[3])
    for tp in tps:
        print(tp.shape)

def test_pun():
    """
    test code
    :return:
    """
    dir = "../../../dl_response_rater/data/pun_of_day/"
    ta_csv = dir + "train1.csv"
    ts_csv = dir + "test1.csv"
    tps = read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)
    for tp in tps:
        print(tp.shape)


def check_argu():
    dir = "../../../dl_response_rater/data/Argu/csv/"
    ta_csv = dir + "generic_VC048263_training.csv"
    ts_csv = dir + "generic_VC048263_testing.csv"
    read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)
    # nb_words will be 10000 and maxlen will be 35

def check_ted():
    dir = "../../../dl_response_rater/data/TED/"
    ta_csv = dir + "train1.csv"
    ts_csv = dir + "test1.csv"
    read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)
    # nb_words 14100 and maxlen 34

def check_asap():
    dir = "../../../dl_response_rater/data/asap2/"
    items = range(1,11)
    for item in items:
        print('item {}'.format(item))
        ta_csv = dir + "train" + str(item) + ".csv"
        ts_csv = dir + "test" + str(item) + ".csv"
        read_input_csv(ta_csv, ts_csv, nb_words=20000, maxlen=30)
    #
    # item     num_wds      len
    # 1        2907         80
    # 2        2720         88
    # 3        2715         66
    # 4        3638         60
    # 5        3550         52
    # 6        3751         54
    # 7        3803         75
    # 8        3803         94
    # 9        4470         97
    # 10       2955         79
if __name__ == '__main__':
    # test_pun()
    # check_argu()
    # check_ted()
    # check_asap()
    test_pun2()