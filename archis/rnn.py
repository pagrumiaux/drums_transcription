#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:48:27 2018

@author: grumiaux
"""

# =============================================================================
# import keras
# from keras.models import Sequential
# from keras.layers import GRU, Bidirectional, Dense
# from keras.optimizers import RMSprop
# import keras.backend as K
# =============================================================================
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import postProcessing
import matplotlib.pyplot as plt
from math import fmod

#%%
params = {'dim_x': 100,
          'dim_y': 168,
          'batch_size': 8,
          'shuffle': True,
          'task': 'RNN',
          'sequential_frames': 100,
          'beatsAndDownbeats': False, 
          'multiTask': True,
          'difference_spectrogram': True}

dataFilter = 'rbma'

#%% Dataset load
dataset = Dataset()
dataset.loadDataset(enst_solo = False)

#%% all IDs
list_IDs = dataset.generate_IDs(params['task'], sequential_frames = params['sequential_frames'], dataFilter=dataFilter)

# IDs three-fold repartition
train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_train_IDs']]
test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_test_IDs']]

n_train_IDs = int(0.85*len(train_IDs))

training_IDs = train_IDs[:n_train_IDs]
validation_IDs = train_IDs[n_train_IDs:]

#%% Model initialization
#units = 30
units = 50
input_shape = (params['dim_x'], params['dim_y'])

model = Sequential()
model.add(Bidirectional(GRU(units, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(GRU(units, return_sequences=True)))
#model.add(Bidirectional(GRU(units, return_sequences=True)))
if not params['multiTask']:
    model.add(Dense(3, activation='sigmoid'))
else:
    model.add(Dense(5, activation='sigmoid'))
optimizer = RMSprop(lr=0.007)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

LRPlateau = ReduceLROnPlateau(factor=0.5, verbose=1, patience=5)
Checkpoint = ModelCheckpoint("SMT-BGRUa-2.{val_loss:.2f}.hdf5", verbose=1)

#%% Model training
patience = 5
epochs = 10

cur_val_loss = 0
best_val_acc = 0 # for refinement
no_improv_count = 0 # count until refinement
for i in range(epochs):
    print(" ")
    print("=== Epoch n°" + str(i) + " ===")
    print("Learning rate : " + str(K.get_value(optimizer.lr)))
    print("Best val acc so far : " + str(best_val_acc))
    print("No improvement count : " + str(no_improv_count))
    
    np.random.shuffle(train_IDs)
    training_IDs = train_IDs[:n_train_IDs]
    validation_IDs = train_IDs[n_train_IDs:]
    training_generator = DataGenerator(**params).generate(dataset, training_IDs)
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

    model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1, callbacks=[LRPlateau])
    
    # check validation accuracy evolution for refinement
    cur_val_loss, cur_val_acc = model.evaluate_generator(validation_generator, steps=len(validation_IDs)//params['batch_size'])
    if cur_val_acc > best_val_acc:
        best_val_acc = cur_val_acc
        no_improv_count = 0
    else:
        if no_improv_count < patience:
            no_improv_count = no_improv_count + 1
        else:
            print("Learning rate decreased")
            cur_lr = K.get_value(optimizer.lr)
            K.set_value(optimizer.lr, cur_lr/3.)
            no_improv_count = 0
    
    print("---> val loss: " + str(cur_val_loss) + " ; val acc: " + str(cur_val_acc))
    
#%% Generate test prediction
    
# we remove the overlapping IDs
#test_IDs_no_overlap = [i for i in test_IDs if fmod(i[1], params['sequential_frames']) == 0]


X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y']))
for i in range(len(test_IDs)):
    X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])

y_hat = model.predict(X_test, verbose=1)

#%%
peak_thres = 0.2
rec_half_length = 0

y_hat_grouped, test_track_IDs = postProcessing.groupePredictionSamplesByTrack(y_hat, list_IDs)
BD_results, SD_results, HH_results, beats_results, downbeats_results, global_results = postProcessing.computeResults(dataset, y_hat_grouped, test_track_IDs, peak_thres, rec_half_length)
print(np.mean(global_results['fmeasure']))

#%%
test_track_ID = 1 # n° test (see test_track_IDs)
postProcessing.visualizeModelPredictionPerTrack(test_track_ID, dataset, y_hat_grouped, test_track_IDs, BD_results, SD_results, HH_results, global_results)
