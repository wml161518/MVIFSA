# coding: utf-8

import json
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Add, MaxPooling1D, Concatenate, Dot, Flatten
from tensorflow.keras import Model
from keras import backend as K
import os
import tensorflow as tf
from configparser import ConfigParser
from tensorflow.keras.optimizers import Adam
from preprocess import readData
from preprocess import readRelation
from layer.attention import Attention

def ranking_loss(y_true, y_pred):
     return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))

def model_construct():
    # CONFIG
    config = ConfigParser()
    config.read('./config.ini')

    question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
    relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
    relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")

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
            trainable=False,name="sharedEmbd_r_w")
    relation_word_emd = sharedEmbd_r_w(relation_all_input)
    sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")
    relation_emd = sharedEmbd_r(relation_input)
    # 2. Information Fusion layer1
    # question bilstm1
    bilstem_layer_1 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2), name="bilstm_layer1")
    question_bilstm_1 = bilstem_layer_1(question_emd)
    # relation word bilstm1
    relation_word_bilstm_1 = bilstem_layer_1(relation_word_emd)
    # relation bilstm1
    relation_bilstm_1 = bilstem_layer_1(relation_emd)

    # 3. Complex Information Representation Layer
    bilstem_layer_2 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2), name="bilstm_layer2")
    # question bilstm2
    question_bilstm_2 = bilstem_layer_2(question_bilstm_1)
    bilstem_layer_3 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2), name="bilstm_layer3")
    # relation word bilstm3
    relation_word_bilstm_3 = bilstem_layer_3(relation_word_bilstm_1)
    # relation bilstm3
    relation_bilstm_3 = bilstem_layer_3(relation_bilstm_1)

    # 4. Residual Learning Layer
    # question residual connect
    question_connect = Add()([question_bilstm_1, question_bilstm_2])
    question_rl_out = MaxPooling1D(80, padding='same')(question_connect)
    # relation word residual connect
    relation_word_connect = Add()([relation_word_bilstm_1, relation_word_bilstm_3])
    relation_word_rc_out = MaxPooling1D(80, padding='same')(relation_word_connect)
    # relation residual connect
    relation_connect = Add()([relation_bilstm_1, relation_bilstm_3])
    relation_rc_out = MaxPooling1D(80, padding='same')(relation_connect)
    relation_rl_out = Add()([relation_rc_out, relation_word_rc_out])
    # 5. Self-Attention Layer
    self_attention = Attention(80)
    question_sa_out = self_attention(question_rl_out)
    relation_sa_out = self_attention(relation_rl_out)
    # finally COSINE SIMILARITY
    question_flatten = Flatten()(question_sa_out)
    relation_flatten = Flatten()(relation_sa_out)
    result = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten])
    model = Model(inputs=[question_input, relation_input, relation_all_input,], outputs=result)
    model.compile(optimizer=Adam(), loss=ranking_loss)
    return model

if __name__ == '__main__':
    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    neg_num = json.load(open('./neg_number.json', 'r'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    model = model_construct()
    model.load_weights('./model/my_model_weights.h5')
    print(model.summary())

    question_feature = np.load('./test_question_feature.npy')

    relation_feature = np.load('./test_relation_feature.npy')
    relation_all_feature = np.load('./test_relation_all_feature.npy')

    print('positive data loaded...')
    simi_pos = model.predict([question_feature, relation_feature, relation_all_feature], batch_size=1024)

    print('positive similarity computed...')
    np.save('test_pre_pos.npy', simi_pos)

    relation_feature_neg = np.load('./test_relation_feature_neg.npy')
    relation_all_feature_neg = np.load('./test_relation_all_feature_neg.npy')

    print('negtive data loaded...')
    simi_neg = model.predict([question_feature, relation_feature_neg, relation_all_feature_neg], batch_size=1024)

    print('negtive similarity computed...')
    np.save('test_pre_neg.npy', simi_neg)

    acc = np.sum(simi_pos>simi_neg) / simi_pos.shape[0]
    print("relation accurcy: " + str(acc))

    index = 0
    false_list = list()
    true_list = list()
    all_set = set()

    config = ConfigParser()
    config.read('./config.ini')
    data = readData(config.get('pre', 'test_filepath'))
    relation = readRelation(config.get('pre', 'relation_filepath'))
    for num,neg_index in neg_num:
        if np.sum(simi_pos[index: index+num]-simi_neg[index: index+num]<0) > 0:
            false_list.append(neg_index)
            print (simi_pos[index])
            print (np.max(simi_neg[index: index+num]))
            print (len(simi_neg[index: index+num]))
            print (np.argmax(simi_neg[index: index+num]))
            print (simi_neg[index: index+num][np.argmax(simi_neg[index: index+num])])
            print (neg_index)
            print ("")
            pass
        else:
            true_list.append(neg_index)
        index += num
        all_set.add(neg_index)
    print (max(true_list))
    true_list = set([i for i in true_list if i in all_set and i not in false_list])
    print (len(all_set))
    print (len(true_list))
    for i in all_set:
        if i not in true_list:
            print (i)
    print (data[0])
    print (relation[0][1])
    print (relation[1][1])
    print('sentence accurcy: '+str(len(true_list)/len(all_set)))
