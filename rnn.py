#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:48:27 2018

@author: grumiaux
"""

import keras
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense
from keras.optimizers import RMSprop
from dataset import Dataset
from dataGenerator import DataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import keras.backend as K
import postProcessing
import matplotlib.pyplot as plt

#%%
params = {'dim_x': 100,
          'dim_y': 168,
          'batch_size': 8,
          'shuffle': True,
          'task': 'RNN',
          'sequential_frames': 100}
index_first_solo_drums = 131

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], sequential_frames = params['sequential_frames'], dataFilter='rbma')
n_ID = len(list_IDs)

#%% ONLY FOR SMT
ID_first_solo = 0 # find solo drums IDs to put it in training IDs list
for i in range(n_ID):
    if list_IDs[i][0] == index_first_solo_drums:
        ID_first_solo = i
        break

# distribute IDs per track block in training or testing sets
list_mix_first_IDs = [i for i in range(index_first_solo_drums)] # list of IDs[0] of MIX
np.random.shuffle(list_mix_first_IDs) # we shuffle solo drums IDs for picking them randomly

#%%
# SMT SOLO DRUMS OR NOT, COMMENT IN OR OUT
#temp_IDs = list_IDs[ID_first_solo:] #temporary list of IDs (will be for training and validation). We start putting solo drums for training in this list
temp_IDs = []

test_IDs = [] #empty list of IDs for evaluation

while len(temp_IDs) < int(2*n_ID/3): # two third of the IDs will be for training and validation
    temp_IDs = temp_IDs + [ID for ID in list_IDs if ID[0] == list_mix_first_IDs[-1]] #we select all IDs of one song each loop
    list_mix_first_IDs = list_mix_first_IDs[:-1] # we remove this ID[0] of the list
    
for first_ID in list_mix_first_IDs: # we put the rest of IDs in the evaluation IDs list
    test_IDs = test_IDs + [ID for ID in list_IDs if ID[0] == first_ID]

train_valid_split = int(0.85*len(temp_IDs)) # we divide the temporary list into training and validation (85% for training)
training_IDs = temp_IDs[:train_valid_split]
validation_IDs = temp_IDs[train_valid_split:]

##%% Generators
#training_generator = DataGenerator(**params).generate(dataset, training_IDs)
#validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

#%% Model initialization
units = 50
input_shape = (params['dim_x'], params['dim_y'])

model = Sequential()
model.add(Bidirectional(GRU(units, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(GRU(units, return_sequences=True)))
#model.add(Bidirectional(GRU(units, return_sequences=True)))
model.add(Dense(3, activation='sigmoid'))
optimizer = RMSprop(lr=0.007)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

LRPlateau = ReduceLROnPlateau(factor=0.5, verbose=1, patience=10)
Checkpoint = ModelCheckpoint("SMT-BGRUa-2.{val_loss:.2f}.hdf5", verbose=1)

#%% Model training
epochs = 20
for i in range(epochs):
    print("=== Epoch n°" + str(i))
#    np.random.seed(i)
    np.random.shuffle(temp_IDs)
    training_IDs = temp_IDs[:train_valid_split]
    validation_IDs = temp_IDs[train_valid_split:]
    training_generator = DataGenerator(**params).generate(dataset, training_IDs)    
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

    print(K.get_value(optimizer.lr))
    model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1, callbacks=[LRPlateau])
    
    
#%% Generate test prediction
X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y']))
for i in range(len(test_IDs)):
    X_test[i, :, :] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])

y_hat = model.predict(X_test, verbose=1)

#%% F measure for BD, SD and HH on the test set
peak_thres = 0.2

#BD_fmeasure = []
#SD_fmeasure = []
#HH_fmeasure = []
global_fmeasure = []

y_hat_grouped, test_track_IDs = postProcessing.groupePredictionSamplesByTrack(y_hat, test_IDs)

for i, ID in enumerate(test_track_IDs):
#    print(i, ID)
    # BD events
    BD_est_activation = y_hat_grouped[i][:, 0]
    BD_est_events = postProcessing.activationToEvents(BD_est_activation, peak_thres = peak_thres)
    BD_ref_events = dataset.data['BD_annotations'][ID]
#    print(BD_ref_events)
    
    # SD events
    SD_est_activation = y_hat_grouped[i][:, 1]
    SD_est_events = postProcessing.activationToEvents(SD_est_activation, peak_thres = peak_thres)
    SD_ref_events = dataset.data['SD_annotations'][ID]
#    print(SD_ref_events)
    
    # HH events
    HH_est_activation = y_hat_grouped[i][:, 2]
    HH_est_events = postProcessing.activationToEvents(HH_est_activation, peak_thres = peak_thres)
    HH_ref_events = dataset.data['HH_annotations'][ID]
#    print(HH_ref_events)
    
    # all events
    est_events = np.concatenate((BD_est_events, SD_est_events, HH_est_events))
    est_pitches = np.concatenate((np.ones(len(BD_est_events)), np.ones(len(SD_est_events))*2, np.ones(len(HH_est_events))*3))
    ref_events = np.concatenate((BD_ref_events, SD_ref_events, HH_ref_events))
    ref_pitches = np.concatenate((np.ones(len(BD_ref_events)), np.ones(len(SD_ref_events))*2, np.ones(len(HH_ref_events))*3))
    
    
#    _, _, fmeasure = postProcessing.precisionRecallFmeasure(est_events, ref_events, est_pitches, ref_pitches)
    precision, recall, fmeasure = postProcessing.f_measure(est_events, ref_events, est_pitches, ref_pitches)
#    input("pause")
    global_fmeasure.append(fmeasure)
    
i = 5 # n° test (see test_track_IDs)
print(dataset.data['audio_name'][test_track_IDs[i]], global_fmeasure[i])
# BD
plt.figure(1)
plt.plot(dataset.data['BD_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 0])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 0], peak_thres = peak_thres))
# SD
plt.figure(2)
plt.plot(dataset.data['SD_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 1])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 1], peak_thres = peak_thres))
# HH
plt.figure(3)
plt.plot(dataset.data['HH_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 2])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 2], peak_thres = peak_thres))