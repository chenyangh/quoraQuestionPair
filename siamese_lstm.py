'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7

According to experiments by kagglers, Theano backend with GPU may give bad LB scores while
        the val_loss seems to be fine, so try Tensorflow backend first please
'''

########################################
## import packages
########################################
import os
import numpy as np
import pandas as pd
import pickle

os.environ['KERAS_BACKEND']='tensorflow'
#os.environ['THEANO_FLAGS'] = 'cuda.root=/usr/local/cuda/ ,device=cuda,floatX=float32'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from gensim.models import KeyedVectors
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.wrappers import  Bidirectional
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers.core import Lambda
import sys
import tensorflow as tf





########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 300
num_dense = 200
rate_drop_lstm = 0.5
rate_drop_dense = 0.5

act = 'tanh'
re_weight = False  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                  rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
                                             binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

with open('all', 'r') as f:
    data_1_train, data_2_train, labels_train, data_1_val, data_2_val, labels_val, word_index, test_ids, test_data_1,\
    test_data_2 = pickle.load(f)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.random.uniform(-1, 1, (nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val == 0] = 1.309028344


def cosine_distance(vests):
    x, y = vests
    x1 = tf.nn.l2_normalize(x, dim=1)
    y1 = tf.nn.l2_normalize(y, dim=1)
    tmp1 = tf.multiply(x1, y1)
    tmp2 = tf.reduce_sum(tmp1, axis=1)
    # tmp = (tf.multiply(tmp2, 1.0/x1/y1) + 1) / 2.0
    return tmp2


def cosine_distance2(vests):
    x, y = vests
    tmp1 = tf.norm(x-y, ord=1, axis=1)

    tmp2 = tf.exp(-tmp1)

    return tmp2


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)


distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, y1])
# preds = Dense(1, activation='sigmoid')(distance)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
              outputs=distance)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
model.summary()
hist = model.fit([data_1_train, data_2_train], labels_train, \
                 validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
                 epochs=2, batch_size=200, shuffle=True, \
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################

print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=1024, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=1024, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)