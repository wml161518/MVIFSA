# coding: utf-8

import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Add, MaxPooling1D, Concatenate, Dot, Flatten
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import configparser
from tensorflow.keras.optimizers import Adam
from layer.attention import Attention
import os

def ranking_loss(y_true, y_pred):
    return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# CONFIG
config = configparser.ConfigParser()
config.read('./config.ini')

# INPUT
question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")
relation_all_input_neg = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input_neg")
relation_input_neg = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input_neg")

# 1. Embedding Layer
question_emd = np.load('./question_emd_matrix.npy')
relation_emd = np.load('./relation_emd_matrix.npy')
relation_all_emd = np.load('./relation_all_emd_matrix.npy')

question_emd = Embedding(question_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[question_emd],
        input_length=config.getint('pre', 'question_maximum_length'),
        trainable=False,name="question_emd")(question_input)

sharedEmbd_r_w = Embedding(relation_all_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_all_emd],
        input_length=config.getint('pre', 'relation_word_maximum_length'),
        trainable=True,name="sharedEmbd_r_w")

relation_word_emd = sharedEmbd_r_w(relation_all_input)

sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")

relation_emd = sharedEmbd_r(relation_input)

relation_word_emd_neg = sharedEmbd_r_w(relation_all_input_neg)

relation_emd_neg = sharedEmbd_r(relation_input_neg)

# 2. Information Fusion layer1
# question bilstm1
bilstem_layer_1 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="bilstm_layer1")
question_bilstm_1 = bilstem_layer_1(question_emd)
# relation word bilstm1
relation_word_bilstm_1 = bilstem_layer_1(relation_word_emd)
relation_word_bilstm_neg_1 = bilstem_layer_1(relation_word_emd_neg)
# relation bilstm1
relation_bilstm_1 = bilstem_layer_1(relation_emd)
relation_bilstm_neg_1 = bilstem_layer_1(relation_emd_neg)

# 3. Complex Information Representation Layer
bilstem_layer_2 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="bilstm_layer2")
# question bilstm2
question_bilstm_2 = bilstem_layer_2(question_bilstm_1)
bilstem_layer_3 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="bilstm_layer3")
# relation word bilstm3
relation_word_bilstm_3 = bilstem_layer_3(relation_word_bilstm_1)
relation_word_bilstm_neg_3 = bilstem_layer_3(relation_word_bilstm_neg_1)
# relation bilstm3
relation_bilstm_3 = bilstem_layer_3(relation_bilstm_1)
relation_bilstm_neg_3 = bilstem_layer_3(relation_bilstm_neg_1)

# 4. Residual Learning Layer
# question residual connect
question_connect = Add()([question_bilstm_1, question_bilstm_2])
question_rl_out = MaxPooling1D(80, padding='same')(question_connect)
# relation word residual connect
relation_word_connect = Add()([relation_word_bilstm_1, relation_word_bilstm_3])
relation_word_neg_connect = Add()([relation_word_bilstm_neg_1, relation_word_bilstm_neg_3])
relation_word_rc_out = MaxPooling1D(80, padding='same')(relation_word_connect)
relation_word_neg_rc_out = MaxPooling1D(80, padding='same')(relation_word_neg_connect)
# relation residual connect
relation_connect = Add()([relation_bilstm_1, relation_bilstm_3])
relation_neg_connect = Add()([relation_bilstm_neg_1, relation_bilstm_neg_3])
relation_rc_out = MaxPooling1D(80, padding='same')(relation_connect)
relation_neg_rc_out = MaxPooling1D(80, padding='same')(relation_neg_connect)
relation_rl_out = Add()([relation_rc_out, relation_word_rc_out])
relation_neg_rl_out = Add()([relation_neg_rc_out, relation_word_neg_rc_out])
# 5. Self-Attention Layer
self_attention = Attention(80)
question_sa_out = self_attention(question_rl_out)
relation_sa_out = self_attention(relation_rl_out)
relation_neg_sa_out = self_attention(relation_neg_rl_out)
# finally COSINE SIMILARITY
question_flatten = Flatten()(question_sa_out)
relation_flatten = Flatten()(relation_sa_out)
relation_flatten_neg = Flatten()(relation_neg_sa_out)
result = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten])
result_neg = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten_neg])

out = Concatenate(axis=-1)([result, result_neg])

model = Model(inputs=[question_input, relation_input, relation_all_input,relation_input_neg, relation_all_input_neg ], outputs=out)
model.compile(optimizer=Adam(), loss=ranking_loss)

print(model.summary())
train_question_features = np.load('./train_question_feature.npy')
train_relation_features = np.load('./train_relation_feature.npy')
train_relation_all_features = np.load('./train_relation_all_feature.npy')
train_relation_features_neg = np.load('./train_relation_feature_neg.npy')
train_relation_all_features_neg = np.load('./train_relation_all_feature_neg.npy')
train_labels = np.load('./train_label.npy')
# model fit and save
model.fit([train_question_features, train_relation_features, train_relation_all_features, train_relation_features_neg, train_relation_all_features_neg], train_labels, epochs=1, batch_size=512, shuffle=True)
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

weights_path = os.path.join(model_dir, 'my_model_weights.h5')
# save model weights
model.save_weights(weights_path)
