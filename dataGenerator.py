#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:36:45 2018

@author: grumiaux
"""

import numpy as np
import keras.backend as K

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, dim_x, dim_y, task, batch_size = 8, shuffle = True, context_frames = 25, sequential_frames = 100, difference_spectrogram = True, beatsAndDownbeats = False, multiTask = False):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task = task
        self.context_frames = context_frames
        self.sequential_frames = sequential_frames
        self.diff = difference_spectrogram
        self.beatsAndDownbeats = beatsAndDownbeats
        self.multiTask = multiTask
    
    def generate(self, dataset, list_IDs, soloDrum = None):
        'Generates batches of samples'
        while 1:
            indexes = self.__get_exploration_order(list_IDs)
            
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                X, y = self.__data_generation(dataset, list_IDs_temp, soloDrum = soloDrum)
                
                yield X, y
        
    def generateForTest(self, dataset, list_IDs, soloDrum = None):
        X, y = self.__data_generation(dataset, list_IDs, soloDrum = soloDrum)
        return X, y
               
    def extract_feature(self, dataset, ID):
        audio_ID = ID[0]
        spectro = dataset.data['mel_spectrogram'][audio_ID]
        audio_spectro_length = spectro.shape[1]
        
        # we standardize the spectrogram
        if dataset.data['origin'][audio_ID] == 'rbma':
            mean = np.array(dataset.standardization['rbma_mel_mean'])
            var = np.array(dataset.standardization['rbma_mel_var'])
        elif dataset.data['origin'][audio_ID] == 'smt':
            mean = dataset.standardization['smt_mel_mean']
            var = dataset.standardization['smt_mel_var']
        mean = mean.reshape((mean.shape[0], 1))
        var = var.reshape((mean.shape[0], 1))
        
        # if the network is a CNN
        if self.task == 'CNN':
            padding = int((self.context_frames-1)/2)
            if ID[1] < padding:
                feature = np.concatenate((np.zeros((spectro.shape[0], (padding-ID[1]))), spectro[:, :ID[1]+padding+1]), axis=1)
            elif ID[1] >= audio_spectro_length - padding:
                feature = np.concatenate((spectro[:, ID[1]-padding:], np.zeros((spectro.shape[0], padding-(audio_spectro_length-ID[1])+1))), axis=1)
            else:
                feature = spectro[:, (ID[1]-padding):(ID[1]+padding+1)]
            
            if self.diff == True:
                feature_diff = np.concatenate((np.zeros((feature.shape[0], 1)), np.diff(feature)), axis=1)
                feature_diff = np.clip(feature_diff, a_min=0, a_max=None)
                feature = np.concatenate((feature, feature_diff), axis=0)
                
        # if the network is a RNN
        elif self.task == 'RNN':
            if ID[1] >= audio_spectro_length - self.sequential_frames:
                feature = np.concatenate((spectro[:, ID[1]:], np.zeros((spectro.shape[0], self.sequential_frames-(audio_spectro_length-ID[1])))), axis=1)
            else:
                feature = spectro[:, ID[1]:ID[1]+self.sequential_frames]

            if self.diff == True:
                feature_diff = np.concatenate((np.zeros((feature.shape[0], 1)), np.diff(feature)), axis=1)
                feature_diff = np.clip(feature_diff, a_min=0, a_max=None)
                feature = np.concatenate((feature, feature_diff), axis=0)
            
            if self.beatsAndDownbeats:
                beats_target = dataset.data['beats_target'][audio_ID]
                downbeats_target = dataset.data['downbeats_target'][audio_ID]
                if ID[1] >= audio_spectro_length - self.sequential_frames:
                    beats = np.concatenate((beats_target[ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                    downbeats = np.concatenate((downbeats_target[ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                else:
                    beats = dataset.data['beats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                    downbeats = dataset.data['downbeats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                
                feature = np.concatenate((feature, np.tile(beats.reshape((1, self.sequential_frames)), (84, 1)), np.tile(downbeats.reshape((1, self.sequential_frames)), (84, 1))), axis=0)
#            feature = feature.T
            
        # if the network is a CBRNN
        elif self.task == 'CBRNN':
            padding = int((self.context_frames-1)/2)
            if not self.diff:
                feature = np.empty((self.sequential_frames, spectro.shape[0], self.dim_y))
            elif self.diff:
                feature = np.empty((self.sequential_frames, spectro.shape[0]*2, self.dim_y))
            
            for i in range(self.sequential_frames):
                if ID[1]+i+padding >= audio_spectro_length:
                    if ID[1]+i-padding >= audio_spectro_length:
                        temp_feature = np.zeros((spectro.shape[0], self.dim_y))
                    else:
                        temp_feature = np.concatenate((spectro[:, (ID[1]+i-padding):], np.zeros((spectro.shape[0], padding-(audio_spectro_length-(ID[1]+i))+1))), axis=1)
                elif ID[1]+i < padding :
                    temp_feature = np.concatenate((np.zeros((spectro.shape[0], padding-(ID[1]+i))), spectro[:, :ID[1]+i+padding+1]), axis=1)
                else:
                    temp_feature = spectro[:, (ID[1]+i-padding):(ID[1]+i+padding+1)]
                
                if self.diff == True:
#                    print(temp_feature.shape)
                    temp_feature_diff = np.concatenate((np.zeros((spectro.shape[0], 1)), np.diff(temp_feature)), axis=1)
#                    print(temp_feature_diff.shape, i)
                    temp_feature_diff = np.clip(temp_feature_diff, a_min=0, a_max=None)
                    feature[i, :, :] = np.concatenate((temp_feature, temp_feature_diff), axis=0)
                else:
                    feature[i, :, :] = temp_feature
                    
        # finish standardization
        if self.diff:
            mean = np.concatenate((mean, mean), axis=0)
            var = np.concatenate((var, var), axis=0)
        feature = (feature-mean)/np.sqrt(var)
        
        # transpose if RNN
        if self.task == 'RNN':
            feature = feature.T
        
        return feature            
                
    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)            
        return indexes
    
    def __data_generation(self, dataset, list_IDs_temp, soloDrum = None):
        'Generates data of batch_size samples'
        
        if self.task == 'CNN':
            if not self.multiTask:
                y_dim = 3
            else:
                y_dim = 5
                
            X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
            y = np.empty((self.batch_size, y_dim), dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                X[i, :, :, 0] = self.extract_feature(dataset, ID)
                y[i, 0] = dataset.data['BD_target'][ID[0]][ID[1]]
                y[i, 1] = dataset.data['SD_target'][ID[0]][ID[1]]
                y[i, 2] = dataset.data['HH_target'][ID[0]][ID[1]]
                if self.multiTask:
                    y[i, 3] = dataset.data['beats_target'][ID[0]][ID[1]]
                    y[i, 4] = dataset.data['downbeats_target'][ID[0]][ID[1]]

        elif self.task == 'RNN':
            if not self.multiTask:
                y_dim = 3
            else:
                y_dim = 5
            
            X = np.empty((self.batch_size, self.dim_x, self.dim_y))
            y = np.empty((self.batch_size, self.dim_x, y_dim), dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                audio_spectro_length = dataset.data['mel_spectrogram'][ID[0]].shape[1]
                X[i, :, :] = self.extract_feature(dataset, ID)
                if ID[1] >= audio_spectro_length - self.sequential_frames:
                    y[i, :, 0] = np.concatenate((dataset.data['BD_target'][ID[0]][ID[1]:], np.zeros((self.dim_x-(audio_spectro_length-ID[1])))))
                    y[i, :, 1] = np.concatenate((dataset.data['SD_target'][ID[0]][ID[1]:], np.zeros((self.dim_x-(audio_spectro_length-ID[1])))))
                    y[i, :, 2] = np.concatenate((dataset.data['HH_target'][ID[0]][ID[1]:], np.zeros((self.dim_x-(audio_spectro_length-ID[1])))))
                    if self.multiTask:
                        y[i, :, 3] = np.concatenate((dataset.data['beats_target'][ID[0]][ID[1]:], np.zeros((self.dim_x-(audio_spectro_length-ID[1])))))
                        y[i, :, 4] = np.concatenate((dataset.data['downbeats_target'][ID[0]][ID[1]:], np.zeros((self.dim_x-(audio_spectro_length-ID[1])))))
                else:
                    y[i, :, 0] = dataset.data['BD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    y[i, :, 1] = dataset.data['SD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    y[i, :, 2] = dataset.data['HH_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    if self.multiTask:
                        y[i, :, 3] = dataset.data['beats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                        y[i, :, 4] = dataset.data['downbeats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                
        elif self.task == 'CBRNN':
            if not self.multiTask:
                y_dim = 3
            else:
                y_dim = 5
                
            X = np.empty((self.batch_size, self.sequential_frames, self.dim_x, self.dim_y, 1))
            y = np.empty((self.batch_size, self.sequential_frames, y_dim), dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                audio_spectro_length = dataset.data['mel_spectrogram'][ID[0]].shape[1]
                X[i, :, :, :, 0] = self.extract_feature(dataset, ID)
                if ID[1] >= audio_spectro_length - self.sequential_frames:
                    y[i, :, 0] = np.concatenate((dataset.data['BD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                    y[i, :, 1] = np.concatenate((dataset.data['SD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                    y[i, :, 2] = np.concatenate((dataset.data['HH_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                    if self.multiTask:
                        y[i, :, 3] = np.concatenate((dataset.data['beats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))
                        y[i, :, 4] = np.concatenate((dataset.data['downbeats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(audio_spectro_length-ID[1])))))

                else:
                    y[i, :, 0] = dataset.data['BD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    y[i, :, 1] = dataset.data['SD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    y[i, :, 2] = dataset.data['HH_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    if self.multiTask:
                        y[i, :, 3] = dataset.data['beats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                        y[i, :, 4] = dataset.data['downbeats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]


        if soloDrum is not None:
            if soloDrum == "BD":
                y = y[:, 0:1]
                
            elif soloDrum == "SD":
                y = y[:, 1:2]
                
            elif soloDrum == "HH":
                y = y[:, 2:3]
                
        return X, np.clip(y, 0+np.finfo(float).eps, None)
    

    
