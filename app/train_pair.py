from __future__ import print_function

import os

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils

from model.model import model_selector
from reader.filereader import read_glove_vectors, read_input_data
from reader.csvreader import read_input_csv

from utils import argumentparser

np.random.seed(42)


def main():
    args = argumentparser.ArgumentParser()
    ta_csv = args.data_dir + "train1.csv"
    ts_csv = args.data_dir + "test1.csv"
    train_pair(args, ta_csv, ts_csv)


def train_pair(args, train_csv, test_csv):
    print('Reading word vectors.')
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    print('Found {} word vectors.'.format(len(embeddings_index)))

    print('Processing input data')
    x_train, y_train, x_test, y_test, word_index, = read_input_csv(train_csv,
                                                                   test_csv,
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
    # args.len_labels_index = len(labels_index)
    args.len_labels_index = 2  # fixed for sentiment detection.

    model = model_selector(args, embedding_matrix)

    checkpoint_filepath = os.path.join(args.model_dir, "weights.best.hdf5")
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True)

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    tsb = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=False)

    callbacks_list = [checkpoint, earlystop, tsb]
    model_json = model.to_json()
    with open(os.path.join(args.model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)

    model.fit(x_train, y_train, validation_split=0.1,
              nb_epoch=args.num_epochs, batch_size=args.batch_size, callbacks=callbacks_list)
    classes = earlystop.model.predict_classes(x_test, batch_size=args.batch_size)
    # acc only supports classes
    acc = np_utils.accuracy(classes, np_utils.categorical_probas_to_classes(y_test))
    print('Test accuracy: {}.'.format(acc))


if __name__ == '__main__':
    main()
