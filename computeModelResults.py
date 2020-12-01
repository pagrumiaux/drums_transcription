#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:59:24 2018

@author: grumiaux
"""

import keras
import time
import tensorflow as tf
import sys
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np

# parse arguments
args = sys.argv
model_name = args[1]
y_hat_file = args[2]


# parameters
params = {'dim_x': 168,
          'dim_y': 9,
          'batch_size': 1,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 9,
          'sequential_frames': 100}

dataFilter = 'smt'

# creating dataset
dataset = Dataset()
dataset.loadDataset(spread_length=5)

# three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], dataFilter=dataFilter)
n_ID = len(list_IDs)

# all IDs
list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], dataFilter=dataFilter)

# IDs three-fold repartition
train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_train_IDs']]
test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_test_IDs']]
n_IDs = len(test_IDs)

n_train_IDs = int(0.85*len(train_IDs))
training_IDs = train_IDs[:n_train_IDs]
validation_IDs = train_IDs[n_train_IDs:]

model = keras.models.load_model(model_name)
print(model.summary())

if params['task'] == 'CBRNN':    
    X_test1 = np.empty((n_IDs, params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
    for i in range(n_IDs):
        X_test1[i, :, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
    y_hat1 = model.predict(X_test1, verbose=1)

    np.save(y_hat_file + "1", y_hat1)

    print("y_hat save to " + y_hat_file + "1 and " + y_hat_file + "2")

if params['task'] == 'CNN':
    X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y'], 1))
    for i in range(len(test_IDs)):
        X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])  
    y_hat = model.predict(X_test, verbose=1)

    np.save(y_hat_file, y_hat)
    print("y_hat for CNN saved to " + y_hat_file)

    
if params['task'] == 'RNN':
    X_test = np.empty((len(test_IDs), params['sequential_frames'], params['dim_y']))
    for i in range(len(test_IDs)):
        X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])
    
    y_hat = model.predict(X_test, verbose=1)
    np.save(y_hat_file, y_hat)
    print("y_hat for RNN saved to " + y_hat_file)

