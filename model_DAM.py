# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 06:47:35 2020

@author: Snehashis
"""


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GRU, LSTM, Input
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import GlobalAveragePooling1D, RepeatVector, multiply, Reshape,Permute, GlobalMaxPooling1D
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten, Reshape, GRU, LSTM, Concatenate, Input, TimeDistributed
from keras.layers.convolutional import *
import keras.backend as K



def model_LSTM_RGB():
    timesteps=7
    data_dim=1024
    print("Create RGB Model")
    input = Input(shape=(timesteps, data_dim))
    LSTM_1 = LSTM(units=1024, activation='tanh', return_sequences=False)(input)
    Res_LSTM_model = Model(inputs=input, outputs=LSTM_1)
    return Res_LSTM_model

def model_input_RGB():
    timesteps=7
    data_dim=1024
    print("Create Attention Model")
    input = Input(shape=(timesteps, data_dim))
    model = Model(inputs=input, outputs=input)
    return model
def attention_reg(weight_mat):
    return 0.0001*K.square((1-K.sum(weight_mat)))




def DAM_Cross_Covariance(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    input_RGB_1 = model_input_RGB()
    input_RGB_2 = model_input_RGB()
    input_structure_1 = input_RGB_1.output
    input_structure_2 = input_RGB_2.output
    cov_temp = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],2,keepdims=True), tensors[1]-K.mean(tensors[1],2,keepdims=True),axes=(2,2))/7,name='Cross_cocariance_temporal')([input_structure_1, input_structure_2])
    cov_temp = Lambda(lambda tensors:tensors * K.eye(K.int_shape(tensors)[-1]))(cov_temp)
    cov_temp = GlobalMaxPooling1D()(cov_temp)
    cov_channel = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],1,keepdims=True), tensors[1]-K.mean(tensors[1],1,keepdims=True),axes=(1,1))/1024,name='Cross_cocariance_channel')([input_structure_1, input_structure_2])
    cov_channel = Lambda(lambda tensors:tensors * K.eye(K.int_shape(tensors)[-1]))(cov_channel)
    cov_channel = GlobalMaxPooling1D()(cov_channel)
    cov_temp = RepeatVector(1024)(cov_temp)
    cov_temp = Permute((2,1))(cov_temp)
    cov_channel = RepeatVector(7)(cov_channel)
    dist = multiply([cov_temp, cov_channel])
    LSTM_1 = LSTM(units=1024,activity_regularizer = l2(0.01),name='Spatial_Attentation',return_sequences=False)(dist)
    spatial_attn = keras.layers.Activation('softmax')(LSTM_1)
    LSTM_1 = keras.layers.Activation('tanh')(LSTM_1)
    temporal_attn = Dense(1, init='glorot_normal', activity_regularizer = l2(0.01),name='RGB_Attentation_1', activation='sigmoid')(LSTM_1)
    spatial_attn_multiply = multiply([model_RGB.output,spatial_attn])
    spatial_attn_add = keras.layers.add([spatial_attn_multiply,model_RGB.output])
    d1 = Dropout(0.2)(spatial_attn_add)
    d1 = Dense(n_neuron, init='glorot_normal', W_regularizer=l2(0.001),activation='relu')(d1)
    detection_output = Dense(1, init='glorot_normal', W_regularizer=l2(0.001),name='Detection_part', activation='sigmoid')(d1)
    Detection_attn_multiply = keras.layers.multiply([detection_output,temporal_attn],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])
    MIL = Model(inputs=[model_RGB.input, input_RGB_1.input, input_RGB_2.input], outputs=[Detection_attn_add])
    return MIL


def DAM_Manhattan(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    input_RGB_1 = model_input_RGB()
    input_RGB_2 = model_input_RGB()
    input_structure_1 = input_RGB_1.output
    input_structure_2 = input_RGB_2.output
    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([input_structure_1, input_structure_2])
    dist1 = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    dist1 = RepeatVector(7)(dist1)
    dist2 = Lambda(lambda tensors:K.sum(tensors, axis=2, keepdims=False))(Similarity)
    dist2 = RepeatVector(1024)(dist2)
    dist2 = Permute((2,1))(dist2)
    dist = multiply([dist1,dist2])
    LSTM_1 = LSTM(units=1024,activity_regularizer = l2(0.01),name='Spatial_Attentation',return_sequences=False)(dist)
    spatial_attn = keras.layers.Activation('softmax')(LSTM_1)
    LSTM_1 = keras.layers.Activation('tanh')(LSTM_1)
    temporal_attn = Dense(1, init='glorot_normal', activity_regularizer = l2(0.01),name='RGB_Attentation_1', activation='sigmoid')(LSTM_1)
    spatial_attn_multiply = multiply([model_RGB.output,spatial_attn])
    spatial_attn_add = keras.layers.add([spatial_attn_multiply,model_RGB.output])
    d1 = Dropout(0.2)(spatial_attn_add)
    d1 = Dense(n_neuron, init='glorot_normal', W_regularizer=l2(0.001),activation='relu')(d1)
    detection_output = Dense(1, init='glorot_normal', W_regularizer=l2(0.001),name='Detection_part', activation='sigmoid')(d1)
    Detection_attn_multiply = keras.layers.multiply([detection_output,temporal_attn],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])
    MIL = Model(inputs=[model_RGB.input, input_RGB_1.input, input_RGB_2.input], outputs=[Detection_attn_add])
    return MIL


