# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:10:44 2020

@author: Snehashis#
"""

import os
import sys
import numpy as np
from keras import backend as K
print(K.backend())
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import plot_model
import multiprocessing
from sklearn.metrics import roc_auc_score, roc_curve,auc
from DL_TRAIN import *
from DL_TEST import *
from model_DAM import *
from RankingLoss import *

seed = 7
np.random.seed(seed)

run_type = "DAM-Dist" # "DAM-Cov"
segment_size = 32
lrn = 0.0001
l3 = 0.001
nuron = 96
dataset = "ST" #UCF


class Metrics_UCF(Callback):
    def on_train_begin(self, logs={}):
        self.ALL_AUC = []
        self.ALL_EPOCH = []
        
    def get_GT(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        # class_folder = test_file[1]
        s1 = int(test_file[2])
        e1 = int(test_file[3])
        s2 = int(test_file[4])
        e2 = int(test_file[5])
        Test_video_name = np.load("./data/Test_video_name_UCF.npy")
        Test_frame_number = np.load("./data/Test_frame_number_UCF.npy")
        video_index = int(np.argwhere(Test_video_name==video_name))
        gt = np.zeros((Test_frame_number[video_index], 1)) 

        if s1 != -1 and e1 != -1:
            gt[s1:e1, 0] = 1
        if s2 != -1 and e2 != -1:
            gt[s2:e2, 0] = 1
        return gt

 
    def my_metrics_DETECTION(self):
        test_gen= DataLoader_Test_UCF(segment_size)
        model_detect = Model(inputs=model.input,outputs=model.get_layer('Detection_part').output)
        score = model_detect.predict_generator(test_gen)

        ############ DETECTION PART ###########

        detection_score=np.asarray(score)
        temp_annotation = './data/Temporal_Anomaly_Annotation_for_Testing_Videos_UCF.txt'
        test_files = [i.strip() for i in open(temp_annotation).readlines()]
        ALL_GT = np.array([])
        ALL_score = np.array([])
        for i in range(len(test_files)):
            video_file = test_files[i]
            video_file = video_file.split()
            video_name = video_file[0][:-4]

            video_segment_score = detection_score[i*segment_size: (i+1)*segment_size]
            video_GT = self.get_GT(test_files[i])
            video_score = np.array([])
            for k in range(segment_size):
                dummy_score = np.repeat(video_segment_score[k, 0], np.floor(video_GT.shape[0] / segment_size))
                video_score = np.concatenate([video_score, dummy_score])

            if video_GT.shape[0] % segment_size != 0:
                dummy_remain_score = np.repeat(video_segment_score[segment_size-1, 0],
                                               video_GT.shape[0] - np.floor(video_GT.shape[0] / segment_size) * segment_size)
                video_score = np.concatenate([video_score, dummy_remain_score])

            video_GT = np.squeeze(video_GT, axis=1)
            ALL_GT = np.concatenate([ALL_GT, video_GT])
            ALL_score = np.concatenate([ALL_score, video_score])

        AUC = roc_auc_score(ALL_GT, ALL_score)
        return AUC


    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 10== 0:
            AUC = self.my_metrics_DETECTION()
            print("###############  AUC : " + str(AUC))
            self.ALL_EPOCH.append(epoch)
            self.ALL_AUC.append(AUC)
            AUC_new = np.asarray(self.ALL_AUC)
            EPOCH_new = np.asarray(self.ALL_EPOCH)
            total = np.asarray(list(zip(EPOCH_new, AUC_new)))
            np.savetxt("./AUC/AUC_"+str(segment_size)+"_Segment_"+run_type+".txt", total, delimiter=',')
            prev_auc = np.max(np.array(AUC_new))
            if AUC >= prev_auc:
                model.save('./model/Model_Weight_'+str(segment_size)+'_Segment_'+run_type+'.h5')
                print('MIL model weights saved Sucessfully')

            print("###############  MAXIMUM AUC : " + str(prev_auc))

        return


class Metrics_ST(Callback):

    def on_train_begin(self, logs={}):
        self.ALL_AUC = []
        self.ALL_EPOCH = []
        self.video_path_normal = './data/Normal_Test_ST.txt'
        self.video_path_anomaly ='./data/Anomaly_Test_ST.txt'
        self.normal_files=[i.strip() for i in open(self.video_path_normal).readlines()]
        self.anomaly_files=[i.strip() for i in open(self.video_path_anomaly).readlines()]

        
    def get_GT(self, video_name):

        if video_name in self.normal_files:
            Test_video_name = np.load("./data/Test_video_name_ST.npy")
            Test_frame_number = np.load("./data/Test_frame_number_ST.npy")
            video_index = int(np.argwhere(Test_video_name==video_name))
            gt = np.zeros((Test_frame_number[video_index], 1))
        elif video_name in self.anomaly_files:
            GT_path = './data/test_frame_mask/' 
            gt = np.load(GT_path + video_name + '.npy')
            gt = np.reshape(gt, (gt.shape[0], 1))
        return gt


    def my_metrics_DETECTION(self):
        test_gen= DataLoader_Test_ST(segment_size)
        model_detect = Model(inputs=model.input,outputs=model.get_layer('Detection_part').output)
        score = model_detect.predict_generator(test_gen)

        ############ DETECTION PART ###########
        detection_score=np.asarray(score)
        test_files_path = './data/SH_Test.txt'
        test_files = [i.strip() for i in open(test_files_path).readlines()]
        ALL_GT = np.array([])
        ALL_score = np.array([])
        for i in range(len(test_files)):
            video_file = test_files[i]
            video_file = video_file.split()
            video_name = video_file[0][:-4]

            video_segment_score = detection_score[i*segment_size: (i+1)*segment_size]
            video_GT = self.get_GT(test_files[i])
            video_score = np.array([])
            for k in range(segment_size):
                dummy_score = np.repeat(video_segment_score[k, 0], np.floor(video_GT.shape[0] / segment_size))
                video_score = np.concatenate([video_score, dummy_score])

            if video_GT.shape[0] % segment_size != 0:
                dummy_remain_score = np.repeat(video_segment_score[segment_size-1, 0],
                                               video_GT.shape[0] - np.floor(video_GT.shape[0] / segment_size) * segment_size)
                video_score = np.concatenate([video_score, dummy_remain_score])

            video_GT = np.squeeze(video_GT, axis=1)
            ALL_GT = np.concatenate([ALL_GT, video_GT])
            ALL_score = np.concatenate([ALL_score, video_score])

        AUC = roc_auc_score(ALL_GT, ALL_score)
        return AUC


    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 10== 0:
            AUC = self.my_metrics_DETECTION()
            print("###############  AUC : " + str(AUC))
            self.ALL_EPOCH.append(epoch)
            self.ALL_AUC.append(AUC)
            AUC_new = np.asarray(self.ALL_AUC)
            EPOCH_new = np.asarray(self.ALL_EPOCH)
            total = np.asarray(list(zip(EPOCH_new, AUC_new)))
            np.savetxt("./AUC/"+dataset+"_AUC_"+str(segment_size)+"_Segment_"+run_type+".txt", total, delimiter=',')
            prev_auc = np.max(np.array(AUC_new))
            if AUC >= prev_auc:
                model.save('./model/'+dataset+'_Model_Weight_'+str(segment_size)+'_Segment_'+run_type+'.h5')
                print('MIL model weights saved Sucessfully')

            print("###############  MAXIMUM AUC : " + str(prev_auc))

        return




adam = Adam(lr=lrn,beta_1=0.9, beta_2=0.999,decay=0.0)
if run_type == "DAM-Dist":
    model = DAM_Manhattan(l3, nuron)
    model.summary()

elif run_type == "DAM-Cov":
    model = DAM_Cross_Covariance(l3, nuron)
    model.summary()      
        

    
model.compile(loss=custom_objective,optimizer=adam)
print("Starting training...")
model.summary()
num_epoch=150000

########################################## TRAINING #####################################################
if dataset == "UCF":
    stp_epc = int(1600/60)
    metrics = Metrics_UCF()
    train_generator = DataLoader_Train_UCF(segment_size)
    loss = model.fit_generator(train_generator, steps_per_epoch=stp_epc, epochs=num_epoch, verbose=1,use_multiprocessing=False,workers=3,callbacks=[metrics])
elif dataset == "ST":
    stp_epc = int(437/24)
    metrics = Metrics_ST()
    train_generator = DataLoader_Train_ST(segment_size)
    loss = model.fit_generator(train_generator, steps_per_epoch=stp_epc, epochs=num_epoch, verbose=1,use_multiprocessing=False,workers=3,callbacks=[metrics])



