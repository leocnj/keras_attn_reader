from __future__ import print_function

import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.engine import Input, Merge
from keras.callbacks import EarlyStopping

from reader.filereader import read_glove_vectors
from reader.csvreader import read_input_csv

from model.model_tweak import model_selector

from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from utils import argumentparser
import csv


np.random.seed(42)




def pun(args):
    '''
    Data providing function:

    :return:
    '''
    train = args.data_dir + '/val_train.csv'
    test = args.data_dir + '/validation.csv'
    x_train, y_train, x_test, y_test, word_index, nb_classes = read_input_csv(train,
                                                                              test,
                                                                              args.nb_words,
                                                                              args.max_sequence_len)
    embedding_matrix = _gen_embd_matrix(args, word_index)

    return x_train, y_train, x_test, y_test, embedding_matrix, nb_classes


def _gen_embd_matrix(args, word_index):
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    embedding_matrix = np.zeros((args.nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > args.nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def model(params):
    '''
    Defined a model for hyperopt to tweak hyper paramters
    Call _kim_cnn_model to define CNN
    Pass several global variables, x_, y_, args, nb_classses, and embedding_matrix

    :param params: dict defining HP ranges
    :return:
    '''

    _model = model_selector(params, args, nb_classes, embedding_matrix)

    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    callbacks_list = [earlystop]

    _model.fit(x_train, y_train,
              batch_size=params['batch_size'],
              nb_epoch=args.num_epochs,
              verbose=1,
              validation_split=0.1,
              callbacks=callbacks_list)

    score, acc = _model.evaluate(x_test, y_test, verbose=1)
    print('test acc:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def trials2csv(trials, csvfile):

    lst = [(t['misc']['vals'], -t['result']['loss']) for t in trials.trials]
    new = []
    for dict, val in lst:
        dict['val'] = val
        new.append(dict)

    keys = new[0].keys()
    with open(csvfile, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(new)


if __name__ == '__main__':

    args = argumentparser.ArgumentParser()
    if (args.dataset == 'pun'):
        x_train, y_train, x_test, y_test, embedding_matrix, nb_classes = pun(args)
        space = {'optimizer': hp.choice('optimizer', ['adadelta', 'rmsprop']),
                 'batch_size': hp.choice('batch_size', [32, 64]),
                 'filter_size': hp.choice('filter_size', [3, 4, 5]),
                 'nb_filter': hp.choice('nb_filter', [75, 100]),
                 'dropout1': hp.uniform('dropout1', 0.25, 0.75),
                 'dropout2': hp.uniform('dropout2', 0.25, 0.75),
                 'use_embeddings': True,
                 'embeddings_trainable': False,
                 'lstm_hs': hp.choice('lstm_hs', [32, 50, 64])}
        trials = Trials()
        best = fmin(model, space, algo=tpe.suggest, max_evals=args.tweak_max, trials=trials)
        print(best)
        trials2csv(trials, 'pun_hp.csv')
    elif(args.dataset == 'ted'):
        x_train, y_train, x_test, y_test, embedding_matrix, nb_classes = pun(args)
        space = {'optimizer': hp.choice('optimizer', ['adadelta', 'rmsprop', 'adam']),
                 'batch_size': 32,
                 'filter_size': hp.choice('filter_size', [3, 4, 5, 6]),
                 'nb_filter': hp.choice('nb_filter', [50, 75, 100, 125]),
                 'dropout1': hp.choice('dropout1', [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]),
                 'dropout2': hp.choice('dropout2', [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]),
                 'use_embeddings': True,
                 'embeddings_trainable': False}
        trials = Trials()
        best = fmin(model, space, algo=tpe.suggest, max_evals=args.tweak_max, trials=trials)
        print(best)
        trials2csv(trials, 'ted_hp.csv')
    else:
        print('wrong dataset')
