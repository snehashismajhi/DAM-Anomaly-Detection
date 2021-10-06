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





def MIL_model(lam3,n_neuron):
    print("Create Model")
    timesteps=7
    data_dim=1024
    print("Create RGB Model")
    input = Input(shape=(timesteps, data_dim))
    LSTM_1 = LSTM(units=512, activation='tanh', return_sequences=False)(input)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(LSTM_1)
    d1_dropout= Dropout(0.6)(d1)
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1_dropout)
    model = Model(inputs=input, outputs=detection_output)
    return model



def model_LSTM_RGB():
    timesteps=7
    data_dim=1024
    print("Create RGB Model")
    input = Input(shape=(timesteps, data_dim))
    LSTM_1 = LSTM(units=1024, activation='tanh', return_sequences=False)(input)
    Res_LSTM_model = Model(inputs=input, outputs=LSTM_1)
    return Res_LSTM_model

def model_temporal_RGB():
    timesteps=7
    data_dim=1024
    print("Create Attention Model")
    input = Input(shape=(timesteps, data_dim))
    # l2_input = Lambda(lambda x: K.l2_normalize(x,axis=1))(input)
    model = Model(inputs=input, outputs=input)
    return model
def attention_reg(weight_mat):
    return 0.0001*K.square((1-K.sum(weight_mat)))




def MIL_model_manhattan_concat(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output
    
    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist1 = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    dist2 = Lambda(lambda tensors:K.sum(tensors, axis=2, keepdims=False))(Similarity)
    feature = keras.layers.concatenate([model_RGB.output, dist1])
    feature = keras.layers.concatenate([feature, dist2])
    feature = keras.layers.Activation('tanh')(feature)

    d1 = Dropout(0.2)(feature)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(d1)
    
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1)

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[detection_output])
    return MIL

def MIL_model_manhattan_withoutLC(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output

    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist1 = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    dist1 = RepeatVector(7)(dist1)
    dist2 = Lambda(lambda tensors:K.sum(tensors, axis=2, keepdims=False))(Similarity)
    dist2 = RepeatVector(1024)(dist2)
    dist2 = Permute((2,1))(dist2)
    dist = multiply([dist1,dist2])

    LSTM_1 = LSTM(units=1024,activity_regularizer = l2(0.01),name='Spatial_Attentation',return_sequences=False)(dist)
    spatial_attn = keras.layers.Activation('softmax')(LSTM_1)
    spatial_attn_multiply = multiply([model_RGB.output,spatial_attn])
    spatial_attn_add = keras.layers.add([spatial_attn_multiply,model_RGB.output])
    d1 = Dropout(0.2)(spatial_attn_add)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(0.001),activation='relu')(d1)
    detection_output = Dense(1, init='glorot_normal', W_regularizer=l2(0.001),name='Detection_part', activation='sigmoid')(d1)

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[detection_output])
    return MIL

def MIL_model_covariance(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output



    cov_temp = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],2,keepdims=True), tensors[1]-K.mean(tensors[1],2,keepdims=True),axes=(2,2))/7,name='Cross_cocariance_temporal')([temporal_structure_1,temporal_structure_2])
    # cov_temp = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_temp)
    cov_temp = GlobalMaxPooling1D()(cov_temp)
    cov_channel = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],1,keepdims=True), tensors[1]-K.mean(tensors[1],1,keepdims=True),axes=(1,1))/1024,name='Cross_cocariance_channel')([temporal_structure_1,temporal_structure_2])
    # cov_channel = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_channel)
    cov_channel = GlobalMaxPooling1D()(cov_channel)
    cov_temp = RepeatVector(1024)(cov_temp)
    cov_temp = Permute((2,1))(cov_temp)
    cov_channel = RepeatVector(7)(cov_channel)
    # cov_channel = Permute((2,1))(cov_channel)


    dist = multiply([cov_temp, cov_channel])
    # dist = cov_temp

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

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL


def MIL_model_covariance_diagonal(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output



    cov_temp = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],2,keepdims=True), tensors[1]-K.mean(tensors[1],2,keepdims=True),axes=(2,2))/7,name='Cross_cocariance_temporal')([temporal_structure_1,temporal_structure_2])
    # cov_temp = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_temp)
    cov_temp = Lambda(lambda tensors:tensors * K.eye(K.int_shape(tensors)[-1]))(cov_temp)
    cov_temp = GlobalMaxPooling1D()(cov_temp)

    cov_channel = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],1,keepdims=True), tensors[1]-K.mean(tensors[1],1,keepdims=True),axes=(1,1))/1024,name='Cross_cocariance_channel')([temporal_structure_1,temporal_structure_2])
    # cov_channel = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_channel)
    cov_channel = Lambda(lambda tensors:tensors * K.eye(K.int_shape(tensors)[-1]))(cov_channel)
    cov_channel = GlobalMaxPooling1D()(cov_channel)
    cov_temp = RepeatVector(1024)(cov_temp)
    cov_temp = Permute((2,1))(cov_temp)
    cov_channel = RepeatVector(7)(cov_channel)
    # cov_channel = Permute((2,1))(cov_channel)


    dist = multiply([cov_temp, cov_channel])
    # dist = cov_temp

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

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL


def MIL_model_covariance_min_max(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output



    cov_temp = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],2,keepdims=True), tensors[1]-K.mean(tensors[1],2,keepdims=True),axes=(2,2))/7,name='Cross_cocariance_temporal')([temporal_structure_1,temporal_structure_2])
    # cov_temp = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_temp)
    cov_temp_max = GlobalMaxPooling1D()(cov_temp)
    cov_temp = Reshape((7,7,1))(cov_temp)
    cov_temp_min = Lambda(lambda x: -K.pool2d(-x, pool_size=(7,1), strides=(1,1)))(cov_temp)#GlobalMaxPooling1D()(-cov_temp)
    cov_temp_min = Flatten()(cov_temp_min)
    cov_channel = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],1,keepdims=True), tensors[1]-K.mean(tensors[1],1,keepdims=True),axes=(1,1))/1024,name='Cross_cocariance_channel')([temporal_structure_1,temporal_structure_2])
    # cov_channel = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(cov_channel)
    cov_channel_max = GlobalMaxPooling1D()(cov_channel)
    cov_channel = Reshape((1024,1024,1))(cov_channel)
    cov_channel_min = Lambda(lambda x: -K.pool2d(-x, pool_size=(1024,1), strides=(1,1)))(cov_channel)#GlobalMaxPooling1D()(-cov_channel)
    cov_channel_min = Flatten()(cov_channel_min)
    
    cov_temp_max = RepeatVector(1024)(cov_temp_max)
    cov_temp_max = Permute((2,1))(cov_temp_max)
    cov_channel_max = RepeatVector(7)(cov_channel_max)
    
    cov_temp_min = RepeatVector(1024)(cov_temp_min)
    cov_temp_min = Permute((2,1))(cov_temp_min)
    cov_channel_min = RepeatVector(7)(cov_channel_min)
    # cov_channel = Permute((2,1))(cov_channel)


    dist_max = multiply([cov_temp_max, cov_channel_max])
    dist_min = multiply([cov_temp_min, cov_channel_min])
    
    dist = multiply([dist_max, dist_min])

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

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL

def MIL_model_covariance_srijan(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output


    abs_diff = multiply([temporal_structure_1,temporal_structure_2])
    cov_temp = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],2,keepdims=True), tensors[1]-K.mean(tensors[1],2,keepdims=True),axes=(2,2))/7,name='Cross_cocariance_temporal')([temporal_structure_1,temporal_structure_2])
    cov_channel = Lambda(lambda tensors:K.batch_dot(tensors[0]-K.mean(tensors[0],1,keepdims=True), tensors[1]-K.mean(tensors[1],1,keepdims=True),axes=(1,1))/1024,name='Cross_cocariance_channel')([temporal_structure_1,temporal_structure_2])
    
    salient_temp = Lambda(lambda tensors:K.batch_dot(tensors[0], tensors[1],axes=(1,1)))([abs_diff,cov_temp])
    salient_channel = Lambda(lambda tensors:K.batch_dot(tensors[0], tensors[1],axes=(2,2)))([abs_diff,cov_channel])
    salient_temp = Permute((2,1))(salient_temp)

    dist = multiply([salient_temp, salient_channel])
    # dist = cov_temp

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

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL
def MIL_model_manhattan(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output

    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
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

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL



def MIL_model_manhattan_uncleaned(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output
    


    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist1 = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    # dist1 = keras.layers.Activation('softmax')(dist1)
    # dist1 = Dense(96, init='glorot_normal', activation='softmax')(dist1)
    dist1 = RepeatVector(7)(dist1)

    dist2 = Lambda(lambda tensors:K.sum(tensors, axis=2, keepdims=False))(Similarity)
    # dist2 = keras.layers.Activation('softmax')(dist2)
    # dist2 = Dense(96, init='glorot_normal', activation='softmax')(dist2)

    dist2 = RepeatVector(1024)(dist2)
    dist2 = Permute((2,1))(dist2)

    dist = multiply([dist1,dist2])
    # dist = keras.layers.Activation('softmax')(dist)
    # dist = keras.layers.add([dist1,dist])
    # dist = keras.layers.concatenate([dist1, dist2])
    # dist_reshape = keras.layers.Reshape((-1, 7, 1024))(dist)
    # dist = Dropout(0.2)(dist)
    LSTM_1 = LSTM(units=1024,activity_regularizer = l2(0.01),name='Spatial_Attentation',return_sequences=False)(dist)
    spatial_attn = keras.layers.Activation('softmax')(LSTM_1)
    LSTM_1 = keras.layers.Activation('tanh')(LSTM_1)
    # attn_val = LSTM(units=1,activation='sigmoid',activity_regularizer = l2(0.01),return_sequences=False)(dist)
    # LSTM_1 = Dropout(0.2)(LSTM_1)
    # max_dist = GlobalMaxPooling1D()(dist)

    # temporal_attn = LSTM(units=7,activity_regularizer = l2(0.01),activation='softmax',return_sequences=False)(dist)
    # temporal_attn = Lambda(lambda tensor:keras.backend.mean(tensor,axis=1, keepdims=False),name='RGB_Attentation_1')(spatial_attn)
    # attn_d1 = Dense(96, init='glorot_normal',activation='relu')(max_dist)
    temporal_attn = Dense(1, init='glorot_normal', activity_regularizer = l2(0.01),name='RGB_Attentation_1', activation='sigmoid')(LSTM_1)
    # temporal_attn = Lambda(lambda tensor:keras.backend.max(tensor,axis=1, keepdims=False),name='RGB_Attentation_1')(temporal_attn)

    spatial_attn_multiply = multiply([model_RGB.output,spatial_attn])
    spatial_attn_add = keras.layers.add([spatial_attn_multiply,model_RGB.output])
    # temporal_attn_repeat = RepeatVector(1024)(temporal_attn)
    # temporal_attn_flat = Flatten()(temporal_attn_repeat)
    # spatial_temporal_attn_multiply = multiply([spatial_attn_add,temporal_attn_flat])
    # spatial_temporal_attn_add = keras.layers.add([spatial_temporal_attn_multiply,spatial_attn_add])
    
    
    d1 = Dropout(0.2)(spatial_attn_add)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(0.001),activation='relu')(d1)
    # d1 = Dropout(0.2)(d1)
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(0.001),name='Detection_part', activation='sigmoid')(d1)
    
    
    
    # temporal_attn = Lambda(lambda tensor:keras.backend.mean(tensor,axis=1, keepdims=False))(spatial_attn)
    # temporal_attn = keras.layers.Activation('sigmoid')(temporal_attn)
    
    Detection_attn_multiply = keras.layers.multiply([detection_output,temporal_attn],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL



def MIL_model_euclidian(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output
    
    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist1 = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    dist1 = Lambda(lambda tensors:K.sqrt(tensors))(dist1)

    attn_d1 = Dense(96, init='glorot_normal',W_regularizer = l2(0.01),activation='relu')(dist1)
    attn_val = Dense(1,init='glorot_normal',name='RGB_Attentation_1',activity_regularizer = l2(0.01), activation='sigmoid')(attn_d1)
 
 
    attn = RepeatVector(1024)(attn_val)
    attn = Flatten()(attn)

    attn_multiply = multiply([model_RGB.output,attn])
    attn_add = keras.layers.add([attn_multiply,model_RGB.output])
    d1 = Dropout(0.2)(attn_add)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(d1)
    
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1)
    
    
    Detection_attn_multiply = keras.layers.multiply([detection_output,attn_val],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL


def MIL_model_manhattan_SSTA(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output
    
    Similarity = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity)
    # dist2 = Lambda(lambda tensors:K.sum(tensors, axis=2, keepdims=False))(Similarity)
    # dist = keras.layers.concatenate([dist1, dist2])
    
    # Similarity_attn = keras.layers.Activation('softmax')(Similarity)
    attn_d1 = Dense(96, init='glorot_normal',W_regularizer = l2(0.01),activation='relu')(dist)
    attn_val1 = Dense(1024,init='glorot_normal',name='RGB_Attentation_1',activity_regularizer = l2(0.01), activation='softmax')(attn_d1)
    
    attn_d2 = Dense(96, init='glorot_normal',W_regularizer = l2(0.01),activation='relu')(dist)
    attn_val2 = Dense(1,init='glorot_normal',name='RGB_Attentation_2',activity_regularizer = l2(0.01), activation='sigmoid')(attn_d2)
    # attn_mul = keras.layers.multiply([attn_val1, attn_val2])
    # attn_val = keras.layers.add([attn_val1,attn_mul])
    
    
    Temporal_attn = RepeatVector(1024)(attn_val2)
    Temporal_attn = Flatten()(Temporal_attn)

    Channel_attn_multiply = multiply([model_RGB.output,attn_val1])
    Channel_attn_add = keras.layers.add([Channel_attn_multiply,model_RGB.output])
    
    Temporal_attn_multiply = multiply([Channel_attn_add,Temporal_attn])
    Temporal_attn_add = keras.layers.add([Channel_attn_add,Temporal_attn_multiply])
    d1 = Dropout(0.2)(Temporal_attn_add)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(d1)
    
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1)
    
    
    Detection_attn_multiply = keras.layers.multiply([detection_output,attn_val2],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL








def MIL_model_manhattan_euclidian(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    temporal_RGB_1 = model_temporal_RGB()
    temporal_RGB_2 = model_temporal_RGB()
    temporal_structure_1 = temporal_RGB_1.output
    temporal_structure_2 = temporal_RGB_2.output
    
    ############################## MANHATTAN DISTANCE ##########################    
    Similarity_MANHAT = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist_MANHAT = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity_MANHAT)
    
    ############################## EUCLIDIAN DISTANCE ###########################
    Similarity_EUCLID = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))([temporal_structure_1,temporal_structure_2])
    dist_EUCLID = Lambda(lambda tensors:K.sum(tensors, axis=1, keepdims=False))(Similarity_EUCLID)
    dist_EUCLID = Lambda(lambda tensors:K.sqrt(tensors))(dist_EUCLID)
    
    dist = keras.layers.concatenate([dist_MANHAT, dist_EUCLID])
    
    
    # Similarity_attn = keras.layers.Activation('softmax')(Similarity)
    # attn_d1 = Dense(256, init='glorot_normal',W_regularizer = l2(0.01),activation='relu')(dist)
    attn_d1 = Dense(256, init='glorot_normal',W_regularizer = l2(0.01),activation='relu')(dist)
    attn_val = Dense(1,init='glorot_normal',name='RGB_Attentation',W_regularizer = l2(0.01), activation='sigmoid')(attn_d1)
    
    # LSTM_2 = LSTM(units=1024, activation='tanh', return_sequences=False)(Similarity_MANHAT)
    # feature = keras.layers.concatenate([model_RGB.output, dist])
    
    attn = RepeatVector(1024)(attn_val)
    attn = Flatten()(attn)
    attn_multiply = multiply([model_RGB.output,attn])
    attn_add = keras.layers.add([attn_multiply,model_RGB.output])
    d1= Dense(96, init='glorot_normal', W_regularizer=l2(lam3))(attn_add)
    d1 = Dropout(0.2)(d1)
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1)
    
    
    Detection_attn_multiply = keras.layers.multiply([detection_output,attn_val],name='Weighted_Detection_part')
    Detection_attn_add = keras.layers.add([detection_output, Detection_attn_multiply])

    MIL = Model(inputs=[model_RGB.input,temporal_RGB_1.input, temporal_RGB_2.input], outputs=[Detection_attn_add])
    return MIL































# from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
# from keras.regularizers import l2
# import keras
# from keras import backend as K
# from keras.models import Sequential, Model
# from keras.layers import Activation, Embedding
# from keras.layers.core import Dense, Flatten, Lambda

# from keras.layers.convolutional import *
# import tensorflow as tf



# def Network():
#     print("Create Model-1")
#     timesteps=32
#     data_dim=7168
#     input = Input(shape=(timesteps, data_dim))
#     # data_dim=1024
#     # input = Input(shape=(1,data_dim))
#     LSTM_1 = LSTM(units=1024, activation='tanh', return_sequences=True)(input)
#     # D_n1 = Dense(512, init='glorot_normal', W_regularizer=l2(0.001))(input)
#     # d1= Dense(96, init='glorot_normal', W_regularizer=l2(0.001), activation='relu')(LSTM_1)
#     network_model = Model(inputs=input, outputs=LSTM_1)
#     return network_model



# def MIL_model():
#     n1 = Network()
#     n2 = Network()
#     n1_output = n1.output
#     n2_output = n2.output
#     # Add a customized layer to compute the absolute difference between the encodings
#     # L1_layer = Lambda(lambda tensors:tf.math.abs([n1.output,n2.output]))
#     L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))([n1_output,n2_output])
#     # L1_distance = L1_layer([encoded_l, encoded_r])



# # 
#     d1= Dense(96, init='glorot_normal', W_regularizer=l2(0.001),activation='relu')(L1_layer)
#     # d1_dropout= Dropout(0.6)(d1)
#     detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(0.001),name='Detection_part', activation='sigmoid')(d1)
#     model = Model(inputs=[n1.input, n2.input], outputs=detection_output)
#     return model