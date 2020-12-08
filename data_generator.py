# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:14:13 2020

@author: PA
"""

import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 n_bins, 
                 model,
                 list_IDs,
                 dataset,
                 batch_size, 
                 shuffle = True, 
                 context_frames = 25, 
                 sequential_frames = 100, 
                 difference_spectrogram = True, 
                 beats_and_downbeats = False, 
                 multiTask = False, 
                 teacher = None, 
                 multiInput = False,
                 solo_drum = False):
        
        'Initialization'
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model
        self.list_IDs = list_IDs
        self.dataset = dataset
        self.context_frames = context_frames
        self.sequential_frames = sequential_frames
        self.diff = difference_spectrogram
        self.beats_and_downbeats = beats_and_downbeats
        self.multiTask = multiTask
        self.teacher = teacher
        self.multiInput = multiInput
        self.solo_drum = solo_drum
        
    def __len__(self):
        return len(self.list_IDs)//self.batch_size

    
    def __getitem__(self, idx):
        idx_batch = self.list_IDs[idx*self.batch_size: (idx+1)*self.batch_size]
        input_feature, target = self.data_load(idx_batch)
        
        return input_feature, target
    
    def data_load(self, idx_batch, solo_drum = None):
        'Generates data of batch_size samples'
        
        if not self.multiTask:
            target_dim = 3
        else:
            target_dim = 5
        
        if self.model == 'CNN':
            if not self.multiInput:
                input_feature = np.empty((self.batch_size, self.context_frames, self.n_bins, 1))
            else:
                input_feature_1 = np.empty((self.batch_size, self.context_frames, self.n_bins, 1))
                input_feature_2 = np.empty((self.batch_size, 2))
            target = np.empty((self.batch_size, target_dim), dtype=int)
            
            for i, ID in enumerate(idx_batch):
                if not self.multiInput:
                    input_feature[i, :, :, 0] = self.extract_feature(ID)
                else:
                    input_feature_1[i, :, :, 0], input_feature_2[i, :] = self.extract_feature(ID)
                    
                target[i, 0] = self.dataset.data['BD_target'][ID[0]][ID[1]]
                target[i, 1] = self.dataset.data['SD_target'][ID[0]][ID[1]]
                target[i, 2] = self.dataset.data['HH_target'][ID[0]][ID[1]]
                if self.multiTask:
                    target[i, 3] = self.dataset.data['beats_target'][ID[0]][ID[1]]
                    target[i, 4] = self.dataset.data['downbeats_target'][ID[0]][ID[1]]

        elif self.model == 'RNN':           
            input_feature = np.empty((self.batch_size, self.sequential_frames, self.n_bins))
            target = np.empty((self.batch_size, self.sequential_frames, target_dim), dtype=int)
            for i, ID in enumerate(idx_batch):
                spectro_length = self.dataset.data['mel_spectrogram'][ID[0]].shape[1]
                input_feature[i, :, :] = self.extract_feature(ID)
                if ID[1] >= spectro_length - self.sequential_frames:
                    target[i, :, 0] = np.concatenate((self.dataset.data['BD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    target[i, :, 1] = np.concatenate((self.dataset.data['SD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    target[i, :, 2] = np.concatenate((self.dataset.data['HH_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    if self.multiTask:
                        target[i, :, 3] = np.concatenate((self.dataset.data['beats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                        target[i, :, 4] = np.concatenate((self.dataset.data['downbeats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                else:
                    target[i, :, 0] = self.dataset.data['BD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    target[i, :, 1] = self.dataset.data['SD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    target[i, :, 2] = self.dataset.data['HH_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    if self.multiTask:
                        target[i, :, 3] = self.dataset.data['beats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                        target[i, :, 4] = self.dataset.data['downbeats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                
        elif self.model == 'CBRNN':
            if not self.multiInput:
                input_feature = np.empty((self.batch_size, self.sequential_frames, self.context_frames, self.n_bins, 1))
            else:
                input_feature_1 = np.empty((self.batch_size, self.sequential_frames, self.context_frames, self.n_bins, 1))
                input_feature_2 = np.empty((self.batch_size, self.sequential_frames, 2))
            
            target = np.empty((self.batch_size, self.sequential_frames, target_dim), dtype=int)
            for i, ID in enumerate(idx_batch):
                spectro_length = self.dataset.data['mel_spectrogram'][ID[0]].shape[1]
                if not self.multiInput:
                    input_feature[i, :, :, :, 0] = self.extract_feature(ID)
                else:
                    input_feature_1[i, :, :, :, 0], input_feature_2[i, :, :] = self.extract_feature(ID)
                if ID[1] >= spectro_length - self.sequential_frames:
                    target[i, :, 0] = np.concatenate((self.dataset.data['BD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    target[i, :, 1] = np.concatenate((self.dataset.data['SD_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    target[i, :, 2] = np.concatenate((self.dataset.data['HH_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    if self.multiTask:
                        target[i, :, 3] = np.concatenate((self.dataset.data['beats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                        target[i, :, 4] = np.concatenate((self.dataset.data['downbeats_target'][ID[0]][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))

                else:
                    target[i, :, 0] = self.dataset.data['BD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    target[i, :, 1] = self.dataset.data['SD_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    target[i, :, 2] = self.dataset.data['HH_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                    if self.multiTask:
                        target[i, :, 3] = self.dataset.data['beats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
                        target[i, :, 4] = self.dataset.data['downbeats_target'][ID[0]][ID[1]:ID[1]+self.sequential_frames]
        
        # if self.model == 'DNN':
        #     input_feature = np.empty((self.batch_size, self.n_frames))
        #     target = np.empty((self.batch_size, target_dim), dtype=int)
        #     for i, ID in enumerate(idx_batch):
        #         input_feature[i, :] = self.extract_feature(ID)
        #         target[i, 0] = self.dataset.data['BD_target'][ID[0]][ID[1]]
        #         target[i, 1] = self.dataset.data['SD_target'][ID[0]][ID[1]]
        #         target[i, 2] = self.dataset.data['HH_target'][ID[0]][ID[1]]
        #         if self.multiTask:
        #             target[i, 3] = self.dataset.data['beats_target'][ID[0]][ID[1]]
        #             target[i, 4] = self.dataset.data['downbeats_target'][ID[0]][ID[1]]

        if solo_drum is not None:
            if solo_drum == "BD":
                target = target[:, 0:1]                
            elif solo_drum == "SD":
                target = target[:, 1:2]                
            elif solo_drum == "HH":
                target = target[:, 2:3]
                
        if not self.multiInput:
            return np.clip(input_feature, 0+np.finfo(float).eps, None), np.clip(target, 0+np.finfo(float).eps, None)
        else:
            return [np.clip(input_feature_1, 0+np.finfo(float).eps, None), np.clip(input_feature_2, 0+np.finfo(float).eps, None)], np.clip(target, 0+np.finfo(float).eps, None)
        
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.list_IDs)

    def extract_feature(self, ID):
        audio_ID = ID[0]
        spectro = self.dataset.data['mel_spectrogram'][audio_ID]
        spectro_length = spectro.shape[1]
        
        # we standardize the spectrogram
        if self.dataset.data['dataset'][audio_ID] == 'rbma' or self.teacher == 'rbma':
            mean = self.dataset.standardization['rbma_mel_mean']
            var = self.dataset.standardization['rbma_mel_var']
        elif self.dataset.data['dataset'][audio_ID] == 'smt' or self.teacher == 'smt':
            mean = self.dataset.standardization['smt_mel_mean']
            var = self.dataset.standardization['smt_mel_var']
        elif self.dataset.data['dataset'][audio_ID] == 'enst' or self.teacher == 'enst':
            mean = self.dataset.standardization['enst_mel_mean']
            var = self.dataset.standardization['enst_mel_var']
        elif self.dataset.data['dataset'][audio_ID] == 'billboard':
            mean = 0
            var = 1
        
        if type(mean) == np.ndarray and type(var) == np.ndarray:
            mean = mean.reshape((mean.shape[0], 1))
            var = var.reshape((mean.shape[0], 1))
        
        
        # if the network is a DNN
        if self.model == 'DNN':
            feature = spectro[:, ID[1]]
            
            if self.diff == True:
                if ID[1] == 0:
                    feature_diff = np.clip(feature, a_min=0, a_max=None)
                else:
                    feature_diff = feature - spectro[:, ID[1]-1]
                    feature_diff = np.clip(feature_diff, a_min=0, a_max=None)
                feature = np.concatenate((feature, feature_diff), axis=0)
                    
#                mean = np.concatenate((mean, mean), axis=0)
#                var = np.concatenate((var, var), axis=0)
                
            feature = feature.reshape((feature.shape[0], 1))
#            print(feature.shape, mean.shape, var.shape)
#            feature = (feature-mean)/np.sqrt(var)
            
            if self.beats_and_downbeats:
                beats_target = self.dataset.data['beats_target'][audio_ID]
                downbeats_target = self.dataset.data['downbeats_target'][audio_ID]
                beats = beats_target[ID[1]]
                downbeats = downbeats_target[ID[1]]
                
                if not self.multiInput:
                    feature = np.concatenate((feature, 1*beats, 1*downbeats), axis=0)
                else:
                    aux_input = np.concatenate((1*beats, 1*downbeats), axis=0)
            
            feature = feature.reshape((feature.shape[0]))
            
        
        # if the network is a CNN
        if self.model == 'CNN':
            padding = int((self.context_frames-1)/2)
            if ID[1] < padding:
                feature = np.concatenate((np.zeros(((padding-ID[1]), spectro.shape[0])), spectro[:, :ID[1]+padding+1].T), axis=0)
            elif ID[1] >= spectro_length - padding:
                feature = np.concatenate((spectro[:, ID[1]-padding:].T, np.zeros((padding-(spectro_length-ID[1])+1, spectro.shape[0]))), axis=0)
            else:
                feature = spectro[:, (ID[1]-padding):(ID[1]+padding+1)].T
            
            if self.diff == True:
                feature_diff = np.concatenate((np.zeros((1, feature.shape[1])), np.diff(feature, axis=0)), axis=0)
                feature_diff = np.clip(feature_diff, a_min=0, a_max=None)
                feature = np.concatenate((feature, feature_diff), axis=1)
                
#                mean = np.concatenate((mean, mean), axis=0)
#                var = np.concatenate((var, var), axis=0)
#            feature = (feature-mean)/np.sqrt(var)
        
            if self.beats_and_downbeats:
                beats_target = self.dataset.data['beats_target'][audio_ID]
                downbeats_target = self.dataset.data['downbeats_target'][audio_ID]
                if ID[1] < padding:
                    beats = np.concatenate((np.zeros((padding-ID[1])), beats_target[:ID[1]+padding+1]))
                    downbeats = np.concatenate((np.zeros((padding-ID[1])), downbeats_target[:ID[1]+padding+1]))
                elif ID[1] >= spectro_length - padding:
                    beats = np.concatenate((beats_target[ID[1]-padding:], np.zeros((padding-(spectro_length-ID[1])+1))))
                    downbeats = np.concatenate((downbeats_target[ID[1]-padding:], np.zeros((padding-(spectro_length-ID[1])+1))))
                else:
                    beats = self.dataset.data['beats_target'][audio_ID][ID[1]-padding:ID[1]+padding+1]
                    downbeats = self.dataset.data['downbeats_target'][audio_ID][ID[1]-padding:ID[1]+padding+1]
                
                feature = np.concatenate((feature, 1*beats.reshape((1, self.context_frames)), 1*downbeats.reshape((1, self.context_frames))), axis=0)
                
            if self.multiInput:
                beats = self.dataset.data['beats_target'][audio_ID][ID[1]]
                downbeats = self.dataset.data['downbeats_target'][audio_ID][ID[1]]
                aux_input = np.array([beats, downbeats])

        # if the network is a RNN
        elif self.model == 'RNN':
            if ID[1] >= spectro_length - self.sequential_frames:
                feature = np.concatenate((spectro[:, ID[1]:].T, np.zeros((self.sequential_frames-(spectro_length-ID[1]), spectro.shape[0]))), axis=0)
            else:
                feature = spectro[:, ID[1]:ID[1]+self.sequential_frames].T

            if self.diff == True:
                feature_diff = np.concatenate((np.zeros((1, feature.shape[1])), np.diff(feature, axis=0)), axis=0)
                feature_diff = np.clip(feature_diff, a_min=0, a_max=None)
                feature = np.concatenate((feature, feature_diff), axis=1)
                
#                mean = np.concatenate((mean, mean), axis=0)
#                var = np.concatenate((var, var), axis=0)
#            feature = (feature-mean)/np.sqrt(var)
            
            if self.beats_and_downbeats:
                beats_target = self.dataset.data['beats_target'][audio_ID]
                downbeats_target = self.dataset.data['downbeats_target'][audio_ID]
                if ID[1] >= spectro_length - self.sequential_frames:
                    beats = np.concatenate((beats_target[ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    downbeats = np.concatenate((downbeats_target[ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                else:
                    beats = self.dataset.data['beats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                    downbeats = self.dataset.data['downbeats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                
                feature = np.concatenate((feature, 1*beats.reshape((1, self.sequential_frames)), 1*downbeats.reshape((1, self.sequential_frames))), axis=0)
            
        # if the network is a CBRNN
        elif self.model == 'CBRNN':
            padding = int((self.context_frames-1)/2)
            if not self.diff:
                if self.beats_and_downbeats:
                    feature = np.empty((self.sequential_frames, self.context_frames, spectro.shape[0]+2))  
                else:
                    feature = np.empty((self.sequential_frames, self.context_frames, spectro.shape[0]))
            elif self.diff:
#                mean = np.concatenate((mean, mean), axis=0)
#                var = np.concatenate((var, var), axis=0)
                if self.beats_and_downbeats:
                    feature = np.empty((self.sequential_frames, self.context_frames, spectro.shape[0]*2+2))
                else:
                    feature = np.empty((self.sequential_frames, self.context_frames, spectro.shape[0]*2))
            
            for i in range(self.sequential_frames):
                if ID[1]+i+padding >= spectro_length:
                    if ID[1]+i-padding >= spectro_length:
                        temp_feature = np.zeros((self.context_frames, spectro.shape[0]))
                    else:
                        temp_feature = np.concatenate((spectro[:, ID[1]+i-padding:].T, np.zeros((padding-(spectro_length-(ID[1]+i))+1, spectro.shape[0]))), axis=0)
                elif ID[1]+i < padding :
                    temp_feature = np.concatenate((np.zeros((padding-(ID[1]+i), spectro.shape[0])), spectro[:, :ID[1]+i+padding+1].T), axis=0)
                else:
                    temp_feature = spectro[:, (ID[1]+i-padding):(ID[1]+i+padding+1)].T
                
                if self.diff == True:
                    temp_feature_diff = np.concatenate((np.zeros((1, spectro.shape[0])), np.diff(temp_feature, axis=0)), axis=0)
                    temp_feature_diff = np.clip(temp_feature_diff, a_min=0, a_max=None)
                    
                    temp_feature = np.concatenate((temp_feature, temp_feature_diff), axis=1)                  
                
                
#                print(temp_feature.shape, mean.shape, var.shape)
#                temp_feature = (temp_feature-mean)/np.sqrt(var)
                    
                if self.beats_and_downbeats:
                    beats_target = self.dataset.data['beats_target'][audio_ID]
                    downbeats_target = self.dataset.data['downbeats_target'][audio_ID]
                    if ID[1]+i+padding >= spectro_length:
                        if ID[1]+i-padding >= spectro_length:
                            beats = np.zeros((self.context_frames))
                            downbeats = np.zeros((self.context_frames))
                        else:
                            beats = np.concatenate((beats_target[ID[1]+i-padding:], np.zeros((padding-(spectro_length-(ID[1]+i))+1))), axis=0)
                            downbeats = np.concatenate((downbeats_target[ID[1]+i-padding:], np.zeros((padding-(spectro_length-(ID[1]+i))+1))), axis=0)
                    elif ID[1]+i < padding:
                        beats = np.concatenate((np.zeros((padding-(ID[1]+i))), beats_target[:ID[1]+i+padding+1]), axis=0)
                        downbeats = np.concatenate((np.zeros((padding-(ID[1]+i))), downbeats_target[:ID[1]+i+padding+1]), axis=0)
                    else:
                        beats = beats_target[(ID[1]+i-padding):(ID[1]+i+padding+1)]
                        downbeats = downbeats_target[(ID[1]+i-padding):(ID[1]+i+padding+1)]

                    temp_feature = np.concatenate((temp_feature, beats.reshape((1, self.context_frames)), downbeats.reshape((1, self.context_frames))), axis=0)
                
                feature[i, :, :] = temp_feature
                
            if self.multiInput:
                if ID[1]+self.sequential_frames > spectro_length:
                    beats = np.concatenate((self.dataset.data['beats_target'][audio_ID][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                    downbeats = np.concatenate((self.dataset.data['downbeats_target'][audio_ID][ID[1]:], np.zeros((self.sequential_frames-(spectro_length-ID[1])))))
                else:
                    beats = self.dataset.data['beats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                    downbeats = self.dataset.data['downbeats_target'][audio_ID][ID[1]:ID[1]+self.sequential_frames]
                aux_input = np.stack([beats, downbeats], axis=1)
            
        if not self.multiInput:
            return feature   
        else:
            return [feature, aux_input]

    
