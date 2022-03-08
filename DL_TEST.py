import numpy as np
import os
from keras.utils import Sequence,to_categorical
from keras.preprocessing import image
import glob
import h5py
from sklearn.preprocessing import normalize




class DataLoader_Test_UCF(Sequence):

    def __init__(self,segment_size):
        self.batch_size = 10
        self.temp_annotation =  './data/Temporal_Anomaly_Annotation_for_Testing_Videos_UCF.txt'
        self.test_files = [i.strip() for i in open(self.temp_annotation).readlines()]
        self.path_frame_feature= './data/I3D_UCF/'
        self.segment_size = segment_size
    def __len__(self):
        return int(len(self.test_files)/self.batch_size)

    def __getitem__(self, item):
        batch_video=self.test_files[item * self.batch_size : (item+1) * self.batch_size]
        feature = [self._get_feature(i) for i in batch_video]
        feature_shift = [self._get_feature_shift(i) for i in batch_video]
        feature=np.asarray(feature)
        feature = np.reshape(feature, (feature.shape[0]*feature.shape[1],feature.shape[2],feature.shape[3]))
        feature_shift=np.asarray(feature_shift)
        feature_shift = np.reshape(feature_shift, (feature_shift.shape[0]*feature_shift.shape[1],feature_shift.shape[2],feature_shift.shape[3]))
        return [feature, feature, feature_shift]


    def _get_feature(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        class_name = test_file[1]
        if class_name == 'Normal':
            class_name = 'Testing_Normal_Videos_Anomaly'
        f = h5py.File(self.path_frame_feature + class_name + '/' + video_name + '.h5', 'r')
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
    

    def _get_feature_shift(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        class_name = test_file[1]
        if class_name == 'Normal':
            class_name = 'Testing_Normal_Videos_Anomaly'
        f = h5py.File(self.path_frame_feature + class_name + '/' + video_name + '.h5', 'r')
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
        feature_segment = feature_segment[0:self.segment_size-1,:,:]
        feature_segment = np.vstack([zero_stack,feature_segment])
        return feature_segment


class DataLoader_Test_ST(Sequence):

    def __init__(self,segment_size):
        self.batch_size = 1
        self.test_files_path = './data/SH_Test.txt'
        self.test_files = [i.strip() for i in open(self.test_files_path).readlines()]
        self.path_frame_feature='./data/I3D_ST\\'
        self.segment_size = segment_size

    def __len__(self):
        return int(len(self.test_files)/self.batch_size)


    def __getitem__(self, item):
        batch_video=self.test_files[item * self.batch_size : (item+1) * self.batch_size]
        feature = [self._get_feature(i) for i in batch_video]
        feature_shift = [self._get_feature_shift(i) for i in batch_video]
        feature=np.asarray(feature)
        feature = np.reshape(feature, (feature.shape[0]*feature.shape[1],feature.shape[2],feature.shape[3]))
        feature_shift=np.asarray(feature_shift)
        feature_shift = np.reshape(feature_shift, (feature_shift.shape[0]*feature_shift.shape[1],feature_shift.shape[2],feature_shift.shape[3]))
        return [feature, feature, feature_shift]

    def _get_feature(self,video_name):
        f = h5py.File(self.path_frame_feature +video_name+'.h5', 'r')
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
        f = h5py.File(self.path_frame_feature +video_name+'.h5', 'r')
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
        feature_segment = feature_segment[0:self.segment_size-1,:,:]
        feature_segment = np.vstack([zero_stack,feature_segment])
        return feature_segment
