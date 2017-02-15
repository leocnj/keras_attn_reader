"""
Attentive LSTM Reader Model
---------------------------

At a high level, this model reads both the story and the question forwards and backwards, and represents the document as a weighted sum of its token where each individual token weight is decided by an attention mechanism that reads the question.
"""

import os
import sys
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.engine import Input, Merge, merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)
from custom import Reverse, masked_concat, masked_dot, masked_sum

### MODEL

def get_model(
        data_path, #Path to dataset
        lstm_dim, #Dimension of the hidden LSTM layers
        optimizer='rmsprop', #Optimization function to be used
        loss='categorical_crossentropy', #Loss function to be used
        weights_path=None #If specified initializes model with weight file given
        ):

    metadata_dict = {}
    f = open(os.path.join(data_path, 'metadata', 'metadata.txt'), 'r')
    for line in f:
        entry = line.split(':')
        metadata_dict[entry[0]] = int(entry[1])
    f.close()
    story_maxlen = metadata_dict['input_length']
    query_maxlen = metadata_dict['query_length']
    vocab_size = metadata_dict['vocab_size']
    entity_dim = metadata_dict['entity_dim']

    embed_weights = np.load(os.path.join(data_path, 'metadata', 'weights.npy'))
    word_dim = embed_weights.shape[1]

########## MODEL ############

    story_input = Input(shape=(story_maxlen,), dtype='int32', name="StoryInput")

    x = Embedding(input_dim=vocab_size+2,
                  output_dim=word_dim,
                  input_length=story_maxlen,
                  mask_zero=True,
                  weights=[embed_weights])(story_input)

    query_input = Input(shape=(query_maxlen,), dtype='int32', name='QueryInput')

    x_q = Embedding(input_dim=vocab_size+2,
            output_dim=word_dim,
            input_length=query_maxlen,
            mask_zero=True,
            weights=[embed_weights])(query_input)

    concat_embeddings = masked_concat([x_q, x], concat_axis=1)

    lstm = LSTM(lstm_dim, consume_less='gpu')(concat_embeddings)

    reverse_lstm = LSTM(lstm_dim, consume_less='gpu', go_backwards=True)(concat_embeddings)

    merged = merge([lstm, reverse_lstm], mode='concat')

    result = Dense(entity_dim, activation='softmax')(merged)

    model = Model(input=[story_input, query_input], output=result)

    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    print(model.summary())
    return model
