#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:08:14 2018

@author: grumiaux
"""

import sys
import keras
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import os
import time

tic = time.time()
#
args = sys.argv
model_name = args[1]
teacher_name = args[2]

os.environ["CUDA_VISIBLE_DEVICES"]= ""

# parameters
params = {'dim_x': 168,
          'dim_y': 9,
          'batch_size': 1,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 9,
          'sequential_frames': 100,
          'beatsAndDownbeats': False, 
          'multiTask': False,
          'teacher': 'smt'} # used for standardization if so


#Dataset load
dataset = Dataset()
dataset.loadDataset(enst_solo = False)

#%%
# list IDs
dataFilter = 'bb'
list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], sequential_frames = params['sequential_frames'], dataFilter=dataFilter)

list_IDs_per_track = []
for i in range(375,1175):
    list_IDs_per_track.append([ID for ID in list_IDs if ID[0] == i])

#%%

model = keras.models.load_model(model_name)
print(model_name + ' loaded for annotation generation.')
#print(model.summary())

for j, track_IDs in enumerate(list_IDs_per_track):
    audio_name = dataset.data['audio_name'][375+j]
#    print(audio_name)
    n_IDs = len(track_IDs)
    
    if params['task'] == 'CBRNN':        
        X = np.empty((n_IDs, params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
        for i in range(n_IDs):
            X[i, :, :, :, 0] = DataGenerator(**params).extract_feature(dataset, track_IDs[i])  
        y_hat = model.predict(X)

    if params['task'] == 'CNN':
        X_test = np.empty((n_IDs, params['dim_x'], params['dim_y'], 1))
        for i in range(n_IDs):
            X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, track_IDs[i])  
        y_hat = model.predict(X_test)
            
    if params['task'] == 'RNN':
        X = np.empty((n_IDs, params['sequential_frames'], params['dim_y']))
        for i in range(n_IDs):
            X[i, :, :]= DataGenerator(**params).extract_feature(dataset, track_IDs[i])        
        y_hat = model.predict(X)
    
    if params['task'] == 'RNN' or params['task'] == 'CBRNN':
        bd = y_hat[:, :, 0].flatten()
        sd = y_hat[:, :, 1].flatten()
        hh = y_hat[:, :, 2].flatten()
    elif params['task'] == 'CNN':
        bd = y_hat[:, 0]
        sd = y_hat[:, 1]
        hh = y_hat[:, 2]
    
    if not os.path.isdir('./' + teacher_name):
        os.makedirs(teacher_name)
        print(teacher_name + ' folder created.')
    
    np.save('./' + teacher_name + '/' + audio_name + '_BD', bd)
    np.save('./' + teacher_name + '/' + audio_name + '_SD', sd)
    np.save('./' + teacher_name + '/' + audio_name + '_HH', hh)        
    
    print(audio_name + ' done')

toc = time.time()
print('Time : ' + str(toc-tic))