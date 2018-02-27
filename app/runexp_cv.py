from __future__ import print_function

from keras.callbacks import EarlyStopping
from model.model_tweak import model_selector
# from keras.utils.np_utils import accuracy     removed after keras2
from reader.filereader import read_glove_vectors
from reader.csvreader import read_input_csv

from utils import argumentparser
import ml_metrics as metrics
import numpy as np
import pandas as pd

np.random.seed(42)

def accuracy(y_pred, y_act):
    return float(len(y_pred == y_act))/float(len(y_act))

def wt_accuracy(accs, wts):
    return np.average(accs, weights=wts)

def asap(args):

    folds = range(1, 11)
    trains = [args.data_dir + 'train' + str(fold) + '.csv' for fold in folds]
    tests = [args.data_dir + 'test' + str(fold) + '.csv' for fold in folds]
    pairs = zip(trains, tests)

    dpw = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    dp2w =[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    ET_yes = args.embedding_trainable
    # Using params found from tweak_params.py
    if args.exp_name == 'cnn':
        params = {
            1:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': dp2w[5],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 4,
                 'nb_filter': 75},
            2:
                {'optimizer': 'adam',
                 'dropout1': dpw[3],
                 'dropout2': dp2w[1],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 3,
                 'nb_filter': 125},
            3:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': dp2w[0],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 6,
                 'nb_filter': 125},
            4:
                {'optimizer': 'adam',
                 'dropout1': dpw[6],
                 'dropout2': dp2w[8],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 4,
                 'nb_filter': 75},
            5:
                {'optimizer': 'adam',
                 'dropout1': dpw[4],
                 'dropout2': dp2w[4],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 4,
                 'nb_filter': 50},
            6:
                {'optimizer': 'adam',
                 'dropout1': dpw[4],
                 'dropout2': dp2w[0],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 4,
                 'nb_filter': 50},
            7:
                {'optimizer': 'adam',
                 'dropout1': dpw[2],
                 'dropout2': dp2w[8],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 3,
                 'nb_filter': 100},
            8:
                {'optimizer': 'adam',
                 'dropout1': dpw[3],
                 'dropout2': dp2w[1],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 5,
                 'nb_filter': 100},
            9:
                {'optimizer': 'adam',
                 'dropout1': dpw[1],
                 'dropout2': dp2w[2],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 6,
                 'nb_filter': 100},
            10:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': dp2w[4],
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'filter_size': 6,
                 'nb_filter': 75},
        }
    else:
        params = {
            1:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': 0.6661,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 48,
                 'two_LSTM': 0},
            2:
                {'optimizer': 'adam',
                 'dropout1': dpw[3],
                 'dropout2': 0.6997,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 48,
                 'two_LSTM': 0},
            3:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': 0.5202,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 24,
                 'two_LSTM': 0},
            4:
                {'optimizer': 'adam',
                 'dropout1': dpw[6],
                 'dropout2': 0.4539,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 32,
                 'two_LSTM': 0},
            5:
                {'optimizer': 'adam',
                 'dropout1': dpw[4],
                 'dropout2': 0.3261,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 48,
                 'two_LSTM': 0},
            6:
                {'optimizer': 'adam',
                 'dropout1': dpw[4],
                 'dropout2': 0.5407,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 60,
                 'two_LSTM': 0},
            7:
                {'optimizer': 'adam',
                 'dropout1': dpw[2],
                 'dropout2': 0.6515,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 48,
                 'two_LSTM': 0},
            8:
                {'optimizer': 'adam',
                 'dropout1': dpw[3],
                 'dropout2': 0.5392,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 60,
                 'two_LSTM': 0},
            9:
                {'optimizer': 'adam',
                 'dropout1': dpw[1],
                 'dropout2': 0.3935,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 32,
                 'two_LSTM': 0},
            10:
                {'optimizer': 'adam',
                 'dropout1': dpw[0],
                 'dropout2': 0.3436,
                 'use_embeddings': True,
                 'embeddings_trainable': ET_yes,
                 'lstm_hs': 32,
                 'two_LSTM': 0}
        }

    accs, kps, wts, acts, preds = train_cv(args, params, pairs)
    cv_kappa = metrics.mean_quadratic_weighted_kappa(kps)
    print('CV weighted mean kw kappa:', cv_kappa)


def pun(args):

    folds = range(1, 11)
    trains = [args.data_dir + 'train_cv' + str(fold) + '.csv' for fold in folds]
    tests = [args.data_dir + 'test_cv' + str(fold) + '.csv' for fold in folds]
    pairs = zip(trains, tests)

    # Using params found from tweak_params.py
    params = {'optimizer': 'rmsprop',
             'batch_size': 32,
             'filter_size': 3,
             'nb_filter': 100,
             'dropout1': 0.2683,
             'dropout2': 0.4900,
             'use_embeddings': True,
             'embeddings_trainable': False}

    accs, kps, wts, acts, preds = train_cv(args, params, pairs)
    acc_cv = wt_accuracy(accs, wts)
    print('10 fold CV accuracy: {}'.format(acc_cv))
    res_df = pd.DataFrame({'act' : acts, 'pred' : preds})
    res_df.to_csv('pun_res.csv', index=False)

def ted(args):

    folds = range(1, 11)
    trains = [args.data_dir + 'train_cv' + str(fold) + '.csv' for fold in folds]
    tests = [args.data_dir + 'test_cv' + str(fold) + '.csv' for fold in folds]
    pairs = zip(trains, tests)

    # Using params found from tweak_params.py
    params = {'optimizer': 'adam',
             'batch_size': 32,
             'filter_size': 6,
             'nb_filter': 100,
             'dropout1': 0.7,
             'dropout2': 0.35,
             'use_embeddings': True,
             'embeddings_trainable': False}

    accs, kps, wts, acts, preds = train_cv(args, params, pairs)
    acc_cv = wt_accuracy(accs, wts)
    print('10 fold CV accuracy: {}'.format(acc_cv))
    res_df = pd.DataFrame({'act' : acts, 'pred' : preds})
    res_df.to_csv('ted_res.csv', index=False)


def argu(args):

    folds = ['VC048263',
             'VC048408',
             'VC084849',
             'VC084851',
             'VC084853',
             'VC101537',
             'VC101541',
             'VC140094',
             'VC207640',
             'VC248479']

    trains = [args.data_dir + 'generic_' + str(fold) + '_training.csv' for fold in folds]
    tests  = [args.data_dir + 'generic_' + str(fold) + '_testing.csv' for fold in folds]
    pairs = zip(trains, tests)

    accs, kps, wts = train_cv(args, pairs)
    acc_cv = wt_accuracy(accs, wts)
    print('10 fold CV accuracy: {}'.format(acc_cv))


def run_a_model(args, params, embedding_matrix, x_train, y_train, x_test, y_test, train_len, test_len):
    '''
    :return:
    '''
    nb_classes = args.len_labels_index

    model = model_selector(params, args, nb_classes, embedding_matrix)
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    callbacks_list = [earlystop]

    model.fit([x_train, train_len], y_train,
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              verbose=1,
              validation_split=0.1,
              callbacks=callbacks_list)
    y_proba = earlystop.model.predict([x_test, test_len], batch_size=args.batch_size)
    # http://bit.ly/2hFvAjS
    # y_act = probas_to_classes(y_test)
    # y_pred = probas_to_classes(y_proba)
    # http://bit.ly/2sYN7xN
    y_act = y_test.argmax(axis=-1)
    y_pred = y_proba.argmax(axis=-1)
    return y_pred, y_act


def train_cv(args, params, pairs):

    print('Reading word vectors.')
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    print('Found {} word vectors.'.format(len(embeddings_index)))

    accs = []
    kps = [] # for ASAP
    wts = []
    acts = np.zeros(0) 
    preds = np.zeros(0)
    fold = 1
    for(train, test) in pairs:
        print(train + '=>' + test + '...')
        x_train, y_train, x_test, y_test, word_index, nb_classes, train_len, test_len = read_input_csv(train,
                                                                                  test,
                                                                                  args.nb_words,
                                                                                  args.max_sequence_len)
        print('train tensor {}.'.format(x_train.shape))

        print('Preparing embedding matrix.')
        # initiate embedding matrix with zero vectors.
        nb_words = min(args.nb_words, len(word_index))
        embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))

        for word, i in word_index.items():
            if i > nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        args.nb_words = nb_words
        args.len_labels_index = nb_classes

        y_pred, y_act = run_a_model(args, params[fold], embedding_matrix, x_train, y_train, x_test, y_test,
                                    train_len, test_len)

        acc = accuracy(y_pred, y_act)
        print('test accuracy: {}'.format(acc))
        kp_v = metrics.quadratic_weighted_kappa(y_pred, y_act)
        kps.append(kp_v)
        accs.append(acc)
        wts.append(y_pred.shape[0])

        acts = np.concatenate([acts, y_act])
        preds = np.concatenate([preds, y_pred])
        fold = fold + 1

    return (accs, kps, wts, acts, preds)


if __name__ == '__main__':

     args = argumentparser.ArgumentParser()
     if(args.dataset == 'pun'):
         pun(args)
     elif args.dataset == 'ted':
         ted(args)
     elif args.dataset == 'asap':
        asap(args)
     elif args.dataset == 'argu':
         argu(args)
     else:
         print('wrong dataset')
