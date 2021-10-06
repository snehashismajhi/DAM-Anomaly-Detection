import numpy as np
import os
from keras.utils import Sequence,to_categorical
from keras.preprocessing import image
import glob
import h5py
from sklearn.preprocessing import normalize




class DataLoader_Train_UCF(Sequence):

    def __init__(self,segment_size):
        self.batch_size = 10
        self.video_path_normal = 'E:\\Research\\code\\Detection\\Normal.txt'#video_path_normal
        self.video_path_anomaly = 'E:\\Research\\code\\Detection\\Abnormal.txt'#video_path_anomaly
        self.normal_files=[i.strip()[:-4] for i in open(self.video_path_normal).readlines()]
        self.anomaly_files=[i.strip()[:-4] for i in open(self.video_path_anomaly).readlines()]
        self.path_frame_feature="E:\\Research\\Features\\UCF\\I3D_center_crop_GAP\\"
        self.segment_size = segment_size

    def __len__(self):
        return int(len(self.normal_files)/self.batch_size)


    def __getitem__(self, item):
        batch_normal=self.normal_files[item * self.batch_size : (item+1) * self.batch_size]
        batch_anomaly = self.anomaly_files[item * self.batch_size : (item + 1) * self.batch_size]
              
        RGB_train_normal=[self._get_feature(i) for i in batch_normal]
        y_train_normal=np.ones((self.batch_size*self.segment_size,1))
        RGB_train_anomaly= [self._get_feature(i) for i in batch_anomaly]
        y_train_anomaly =np.zeros((self.batch_size*self.segment_size,1))
        
        RGB_train_normal = np.asarray(RGB_train_normal)
        y_train_normal = np.array(y_train_normal)
        RGB_train_anomaly = np.asarray(RGB_train_anomaly)
        y_train_anomaly = np.array(y_train_anomaly)
        
        RGB_train= np.vstack([RGB_train_anomaly,RGB_train_normal])
        RGB_train = np.reshape(RGB_train,(RGB_train.shape[0]*RGB_train.shape[1],RGB_train.shape[2],RGB_train.shape[3]))
        y_train= np.concatenate([y_train_anomaly,y_train_normal])
        
        
        RGB_train_normal_shift=[self._get_feature_shift(i) for i in batch_normal]
        RGB_train_anomaly_shift= [self._get_feature_shift(i) for i in batch_anomaly]
               
        RGB_train_normal_shift = np.asarray(RGB_train_normal_shift)
        RGB_train_anomaly_shift = np.asarray(RGB_train_anomaly_shift)
        
        RGB_train_shift= np.vstack([RGB_train_anomaly_shift,RGB_train_normal_shift])
        RGB_train_shift = np.reshape(RGB_train_shift,(RGB_train_shift.shape[0]*RGB_train_shift.shape[1],RGB_train_shift.shape[2],RGB_train_shift.shape[3]))
        
        return [RGB_train, RGB_train, RGB_train_shift], y_train



    def _get_feature(self,video_name):
        # print(video_name)
        folder=video_name.split('/')
        class_name=folder[0]
        video = folder[1]
        video_path = class_name+'\\'+video
        f = h5py.File(self.path_frame_feature + video_path+'.h5', 'r')
        feature = np.array(f['features'])
        feature=np.squeeze(feature,axis=0)
        feature=np.squeeze(feature,axis=3)
        feature=np.squeeze(feature,axis=2)
        feature_norm = feature

        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                # print(i)
                temp = feature_norm[k]
                temp = np.reshape(temp,(1,temp.shape[0],temp.shape[1]))
                feature_segment = np.vstack([feature_segment,temp])
                i=i+1
        else:
            feature_segment=[]
            for i in range(self.segment_size-1):
                j=int(feature_norm.shape[0]/self.segment_size)
                temp=feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
                
            i = feature_norm.shape[0] - (self.segment_size * j)
            temp=feature_norm[i:feature_norm.shape[0]]
            feature_segment.append(np.max(temp,axis=0))
            feature_segment=np.asarray(feature_segment)            



        return feature_segment
    
    
    
    
    def _get_feature_shift(self,video_name):
        # print(video_name)
        folder=video_name.split('/')
        class_name=folder[0]
        video = folder[1]
        video_path = class_name+'\\'+video
        f = h5py.File(self.path_frame_feature + video_path+'.h5', 'r')
        feature = np.array(f['features'])
        feature=np.squeeze(feature,axis=0)
        feature=np.squeeze(feature,axis=3)
        feature=np.squeeze(feature,axis=2)
        feature_norm = feature
        # zero_stack = feature_norm[0]#np.zeros((1,7,1024))
        # zero_stack = np.reshape(zero_stack,(1,zero_stack.shape[0],zero_stack.shape[1]))

        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                # print(i)
                temp = feature_norm[k]
                temp = np.reshape(temp,(1,temp.shape[0],temp.shape[1]))
                feature_segment = np.vstack([feature_segment,temp])
                i=i+1
        else:
            feature_segment=[]
            for i in range(self.segment_size-1):
                j=int(feature_norm.shape[0]/self.segment_size)
                temp=feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
                
            i = feature_norm.shape[0] - (self.segment_size * j)
            temp=feature_norm[i:feature_norm.shape[0]]
            feature_segment.append(np.max(temp,axis=0))
            feature_segment=np.asarray(feature_segment)    

        zero_stack = np.zeros((1,7,1024))
        # zero_stack = np.reshape(zero_stack,(1,zero_stack.shape[0],zero_stack.shape[1]))
        feature_segment = feature_segment[0:self.segment_size-1,:,:]#np.delete(feature_segment,(self.segment_size-1),axis =0)
        feature_segment = np.vstack([zero_stack,feature_segment])
        # print(feature_segment)
        
        return feature_segment











class DataLoader_Train_ST(Sequence):

    def __init__(self,segment_size):
        self.batch_size = 12
        self.video_path_normal = 'E:\\Research\\code\\ST_Detection\\Normal_train_sub_set_new.txt'
        # self.video_path_normal = 'E:\\Research\\code\\ST_Detection\\Normal_train.txt'
        self.video_path_anomaly ='E:\\Research\\code\\ST_Detection\\Anomaly_train.txt'
        self.normal_files=[i.strip() for i in open(self.video_path_normal).readlines()]
        self.anomaly_files=[i.strip() for i in open(self.video_path_anomaly).readlines()]
        self.path_frame_feature='E:\\Research\\Features\\ST\\3D\\I3D_center\\'   ############ IMAGE NET +KINETICS FEATURE
        self.segment_size = segment_size




    def __len__(self):
        return int(len(self.anomaly_files)/(self.batch_size))


    def __getitem__(self, item):
        batch_normal=self.normal_files[item * self.batch_size : (item+1) * self.batch_size]
        batch_anomaly = self.anomaly_files[item * self.batch_size : (item + 1) * self.batch_size]
              
        RGB_train_normal=[self._get_feature(i) for i in batch_normal]
        y_train_normal=np.ones((self.batch_size*self.segment_size,1))
        RGB_train_anomaly= [self._get_feature(i) for i in batch_anomaly]
        y_train_anomaly =np.zeros((self.batch_size*self.segment_size,1))
        
        RGB_train_normal = np.asarray(RGB_train_normal)
        y_train_normal = np.array(y_train_normal)
        RGB_train_anomaly = np.asarray(RGB_train_anomaly)
        y_train_anomaly = np.array(y_train_anomaly)
        
        RGB_train= np.vstack([RGB_train_anomaly,RGB_train_normal])
        RGB_train = np.reshape(RGB_train,(RGB_train.shape[0]*RGB_train.shape[1],RGB_train.shape[2],RGB_train.shape[3]))
        y_train= np.concatenate([y_train_anomaly,y_train_normal])
        
        
        RGB_train_normal_shift=[self._get_feature_shift(i) for i in batch_normal]
        RGB_train_anomaly_shift= [self._get_feature_shift(i) for i in batch_anomaly]
               
        RGB_train_normal_shift = np.asarray(RGB_train_normal_shift)
        RGB_train_anomaly_shift = np.asarray(RGB_train_anomaly_shift)
        
        RGB_train_shift= np.vstack([RGB_train_anomaly_shift,RGB_train_normal_shift])
        RGB_train_shift = np.reshape(RGB_train_shift,(RGB_train_shift.shape[0]*RGB_train_shift.shape[1],RGB_train_shift.shape[2],RGB_train_shift.shape[3]))
        
        return [RGB_train, RGB_train, RGB_train_shift], y_train




    def _get_feature(self,video_name):
        # print(video_name)
        f = h5py.File(self.path_frame_feature + video_name+'.h5', 'r')
        feature = np.array(f['features'])
        feature=np.squeeze(feature,axis=0)
        feature = np.squeeze(feature,axis=3)
        feature = np.squeeze(feature,axis=2)
        feature_norm = feature
        
        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                # print(i)
                temp = feature_norm[k]
                temp = np.reshape(temp,(1,temp.shape[0],temp.shape[1]))
                feature_segment = np.vstack([feature_segment,temp])
                i=i+1
        else:
            feature_segment=[]
            for i in range(self.segment_size-1):
                j=int(feature_norm.shape[0]/self.segment_size)
                temp=feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
                
            i = feature_norm.shape[0] - (self.segment_size * j)
            temp=feature_norm[i:feature_norm.shape[0]]
            feature_segment.append(np.max(temp,axis=0))
            feature_segment=np.asarray(feature_segment)            

        return feature_segment
    
    
    
    def _get_feature_shift(self,video_name):
        # print(video_name)
        f = h5py.File(self.path_frame_feature + video_name+'.h5', 'r')
        feature = np.array(f['features'])
        feature=np.squeeze(feature,axis=0)
        feature = np.squeeze(feature,axis=3)
        feature = np.squeeze(feature,axis=2)
        feature_norm = feature
        
        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                # print(i)
                temp = feature_norm[k]
                temp = np.reshape(temp,(1,temp.shape[0],temp.shape[1]))
                feature_segment = np.vstack([feature_segment,temp])
                i=i+1
        else:
            feature_segment=[]
            for i in range(self.segment_size-1):
                j=int(feature_norm.shape[0]/self.segment_size)
                temp=feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
                
            i = feature_norm.shape[0] - (self.segment_size * j)
            temp=feature_norm[i:feature_norm.shape[0]]
            feature_segment.append(np.max(temp,axis=0))
            feature_segment=np.asarray(feature_segment)    

        zero_stack = np.zeros((1,7,1024))
        # zero_stack = np.reshape(zero_stack,(1,zero_stack.shape[0],zero_stack.shape[1]))
        feature_segment = feature_segment[0:self.segment_size-1,:,:]#np.delete(feature_segment,(self.segment_size-1),axis =0)
        feature_segment = np.vstack([zero_stack,feature_segment])
        # print(feature_segment)
             

        return feature_segment



