#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:59:24 2018

@author: grumiaux
"""

import keras
import sys
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import keras.backend as K
import manage_gpus as gpl
import tensorflow as tf
import os
import time

# parse arguments
args = sys.argv
model_name = args[1]
y_hat_file = args[2]

os.environ["CUDA_VISIBLE_DEVICES"]= ""
#
#obtain lock
gpu_ids = gpl.board_ids()
if gpu_ids is None:
    gpu_device = None
else:
    gpu_device = -1

gpu_id_locked = -1
if gpu_device is not None:
    gpu_id_locked = gpl.obtain_lock_id(id=gpu_device)
    if gpu_id_locked < 0:
        time.sleep(3)
        if gpu_id_locked < 0:
            if gpu_device < 0:
                raise RuntimeError("No GPUs available for locking")
            else:
                raise RuntimeError("cannot obtain any of the selected GPUs {0}".format(str(gpu_device)))
    comp_device = "/GPU:0"
else:
    comp_device = "/cpu:0"

with tf.device(comp_device):

    # parameters
    params = {'dim_x': 168,
              'dim_y': 25,
              'batch_size': 1,
              'shuffle': True,
              'task': 'CNN',
              'context_frames': 25,
              'sequential_frames': 400,
              'beatsAndDownbeats': False, 
              'multiTask': False,
              'multiInput': False}
    
    dataFilter = 'enst'
    all_ids = True
    
    # Dataset load
    dataset = Dataset()
    dataset.loadDataset(enst_solo = False)
    
    # all IDs
    list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], sequential_frames = params['sequential_frames'], dataFilter=dataFilter)
    
    # IDs three-fold repartition
    train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_train_IDs']]
    test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_test_IDs']]
    
    n_train_IDs = int(0.85*len(train_IDs))
    training_IDs = train_IDs[:n_train_IDs]
    validation_IDs = train_IDs[n_train_IDs:]
    
    K.set_learning_phase(1)
    model = keras.models.load_model(model_name)
    print(model.summary())
    
    if params['task'] == 'CBRNN':
        
        if all_ids:
            X_test = np.empty((len(list_IDs), params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
            for i in range(len(list_IDs)):
                X_test[i, :, :, :, 0] = DataGenerator(**params).extract_feature(dataset, list_IDs[i])  
            y_hat = model.predict(X_test, verbose=1)
        else:
            if not params['multiInput']:
                X_test = np.empty((len(test_IDs), params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
                for i in range(len(test_IDs)):
                    X_test[i, :, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
                y_hat = model.predict(X_test, verbose=1)
            else:
                X1 = np.empty((len(test_IDs), params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
                X2 = np.empty((len(test_IDs), params['sequential_frames'], 2))
                for i in range(len(test_IDs)):
                    X1[i, :, :, :, 0], X2[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
                y_hat = model.predict([X1, X2], verbose=1)
        np.save(y_hat_file, y_hat)
        print("y_hat CBRNN saved to " + y_hat_file)
    
    if params['task'] == 'CNN':
        if all_ids:
            X_test = np.empty((len(list_IDs), params['dim_x'], params['dim_y'], 1))
            for i in range(len(list_IDs)):
                X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, list_IDs[i])  
            y_hat = model.predict(X_test, verbose=1)
        else:
            if not params['multiInput']:
                X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y'], 1))
                for i in range(len(test_IDs)):
                    X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
                y_hat = model.predict(X_test, verbose=1)
            else:
                X1 = np.empty((len(test_IDs), params['dim_x'], params['dim_y'], 1))
                X2 = np.empty((len(test_IDs), 2))
                for i in range(len(test_IDs)):
                    X1[i, :, :, 0], X2[i, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
                y_hat = model.predict([X1, X2], verbose=1)
    
    #    y_hat = model.predict(X_test, verbose=1)
        np.save(y_hat_file, y_hat)
        print("y_hat for CNN saved to " + y_hat_file)
    
        
    if params['task'] == 'RNN':
        if all_ids:
            X_test = np.empty((len(list_IDs), params['sequential_frames'], params['dim_y']))
            for i in range(len(list_IDs)):
                X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, list_IDs[i])
            y_hat = model.predict(X_test, verbose=1)
        else:
            X_test = np.empty((len(test_IDs), params['sequential_frames'], params['dim_y']))
            for i in range(len(test_IDs)):
                X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])
            y_hat = model.predict(X_test, verbose=1)
        
    #    y_hat = model.predict(X_test, verbose=1)
        np.save(y_hat_file, y_hat)
        print("y_hat for RNN saved to " + y_hat_file)
    
    if params['task'] == 'DNN':
        if all_ids:
            X_test = np.empty((len(list_IDs), params['dim_x']))
            for i in range(len(list_IDs)):
                X_test[i, :] = DataGenerator(**params).extract_feature(dataset, list_IDs[i])
        else:
            X_test = np.empty((len(test_IDs), params['dim_x']))
            for i in range(len(test_IDs)):
                X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])
        
        y_hat = model.predict(X_test, verbose=1)
        np.save(y_hat_file, y_hat)
        print("y_hat for RNN saved to " + y_hat_file)
    
if gpu_id_locked >= 0:
    gpl.free_lock(gpu_id_locked)
