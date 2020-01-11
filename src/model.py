# -*- coding:utf-8 -*-
# @Author: pgzhang

from keras.models import model_from_json
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input, Embedding, LSTM
from keras.layers.merge import add
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import os
import keras
from parameters import *


def model(vocab_size:int, max_len:int) -> Model:
    img_input = Input(shape=(224, 224, 3), name='img_input', dtype='float32', sparse=False)
    block1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(
                              scale=1.0, mode='fan_avg', distribution='uniform'),
                          name='block1_conv1')(img_input)
    block1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(
                              scale=1.0, mode='fan_avg', distribution='uniform'),
                          name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2), padding='valid',
                               strides=(2, 2), name='block1_pool')(block1_conv2)


    block2_conv1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(
                              scale=1.0, mode='fan_avg', distribution='uniform'),
                          name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D(strides=(2, 2), name='block2_pool')(block2_conv2)
    

    block3_conv1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block3_conv2')(block3_conv1)
    
    block3_conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D(strides=(2, 2), name='block3_pool')(block3_conv3)
    

    block4_conv1 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D(strides=(2, 2), name='block4_pool')(block4_conv3)
    
    
    block5_conv1 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          activation='relu',
                          kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                distribution='uniform'),
                          name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D(strides=(2, 2), name='block5_pool')(block5_conv3)

    flatten = Flatten(name='flatten')(block5_pool)

    fc1 = Dense(units=4096, activation='relu',
                kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                      distribution='uniform'),
                name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu',
                kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                      distribution='uniform'),
                name='fc2')(fc1)


    dropout_1 = Dropout(0.5)(fc2)
    dense_1 = Dense(256,activation='relu')(dropout_1)

    seq_input = Input(shape=(max_len,),name='seq_input')
    embed_1 = Embedding(vocab_size, 256, mask_zero=True)(seq_input)
    seq_dropout_1 = Dropout(0.5)(embed_1)
    lstm_1 = LSTM(256,activation='relu')(seq_dropout_1)

    decoder_1 = add([dense_1,lstm_1])
    decoder_dense = Dense(256,activation='relu')(decoder_1)

    outputs = Dense(vocab_size,activation='softmax')(decoder_dense)

    model = Model(inputs =[img_input,seq_input],outputs=outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam')

    return model







