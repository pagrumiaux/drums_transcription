#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:54 2018

@author: grumiaux
"""

import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from dataset import Dataset
from dataGenerator import DataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import postProcessing
import utilities

#%%
params = {'dim_x': 168,
          'dim_y': 9,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 9}

index_first_solo_drums = 131

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], dataFilter='rbma')
n_ID = len(list_IDs)

#%%
ID_first_solo = 0 # find solo drums IDs to put it in training IDs list
for i in range(n_ID):
    if list_IDs[i][0] == index_first_solo_drums:
        ID_first_solo = i
        break

#%% distribute IDs per track block in training or testing sets
list_mix_first_IDs = [i for i in range(index_first_solo_drums)] # list of IDs[0] of MIX
np.random.shuffle(list_mix_first_IDs) # we shuffle solo drums IDs for picking them randomly

# comment/uncomment consistent one
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
   
#%%
### model creation ###
input_shape = (params['dim_x'], params['context_frames'], 1)

model = Sequential()
# block A
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))
# block B
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

LRPlateau = ReduceLROnPlateau(factor=0.5, verbose=1, patience=10)
Checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

#%%
patience = 5
epochs = 20

cur_val_loss = 0
best_val_acc = 0 # for refinement
no_improv_count = 0 # count until refinement
for i in range(epochs):
    print(" ")
    print("=== Epoch n°" + str(i) + " ===")
    print("Learning rate : " + str(K.get_value(optimizer.lr)))
    print("Best val acc so far : " + str(best_val_acc))
    print("No improvement count : " + str(no_improv_count))

    np.random.shuffle(temp_IDs)
    training_IDs = temp_IDs[:train_valid_split]
    validation_IDs = temp_IDs[train_valid_split:]
    training_IDs_with_duplicate = utilities.duplicateTrainingSamples(dataset, training_IDs, ratio = 4)
    training_generator = DataGenerator(**params).generate(dataset, training_IDs_with_duplicate)    
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)
    
#    print(K.get_value(optimizer.lr))
    model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs_with_duplicate)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1, callbacks=[LRPlateau])

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

#%%
X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y'], 1))
for i in range(len(test_IDs)):
    X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])
    
y_hat = model.predict(X_test, verbose=1)

#%% F measure for BD, SD and HH on the test set
peak_thres = 0.2

# results dict initialization
BD_results = {'precision': [], 'recall': [], 'fmeasure': []}
SD_results = {'precision': [], 'recall': [], 'fmeasure': []}
HH_results = {'precision': [], 'recall': [], 'fmeasure': []}
global_results = {'precision': [], 'recall': [], 'fmeasure': []}

y_hat_grouped, test_track_IDs = postProcessing.groupePredictionSamplesByTrack(y_hat, test_IDs)

for i, ID in enumerate(test_track_IDs):
    # BD events
    BD_est_activation = y_hat_grouped[i][:, 0]
    BD_est_events = postProcessing.activationToEvents(BD_est_activation, peak_thres = peak_thres)
    BD_ref_events = np.array(dataset.data['BD_annotations'][ID])
    BD_est_pitches = np.ones(len(BD_est_events))
    BD_ref_pitches = np.ones(len(BD_ref_events))
    BD_precision, BD_recall, BD_fmeasure = postProcessing.f_measure(BD_est_events, BD_ref_events, BD_est_pitches, BD_ref_pitches)
    BD_results['precision'].append(BD_precision)
    BD_results['recall'].append(BD_recall)
    BD_results['fmeasure'].append(BD_fmeasure)
  
    # SD events
    SD_est_activation = y_hat_grouped[i][:, 1]
    SD_est_events = postProcessing.activationToEvents(SD_est_activation, peak_thres = peak_thres)
    SD_ref_events = np.array(dataset.data['SD_annotations'][ID])
    SD_est_pitches = np.ones(len(SD_est_events))
    SD_ref_pitches = np.ones(len(SD_ref_events))
    SD_precision, SD_recall, SD_fmeasure = postProcessing.f_measure(SD_est_events, SD_ref_events, SD_est_pitches, SD_ref_pitches)
    SD_results['precision'].append(SD_precision)
    SD_results['recall'].append(SD_recall)
    SD_results['fmeasure'].append(SD_fmeasure)

    # HH events
    HH_est_activation = y_hat_grouped[i][:, 2]
    HH_est_events = postProcessing.activationToEvents(HH_est_activation, peak_thres = peak_thres)
    HH_ref_events = np.array(dataset.data['HH_annotations'][ID])
    HH_est_pitches = np.ones(len(HH_est_events))
    HH_ref_pitches = np.ones(len(HH_ref_events))
    HH_precision, HH_recall, HH_fmeasure = postProcessing.f_measure(HH_est_events, HH_ref_events, HH_est_pitches, HH_ref_pitches)
    HH_results['precision'].append(HH_precision)
    HH_results['recall'].append(HH_recall)
    HH_results['fmeasure'].append(HH_fmeasure)

    # all events
    all_est_events = np.concatenate((BD_est_events, SD_est_events, HH_est_events))
    all_est_pitches = np.concatenate((np.ones(len(BD_est_events)), np.ones(len(SD_est_events))*2, np.ones(len(HH_est_events))*3))
    all_ref_events = np.concatenate((BD_ref_events, SD_ref_events, HH_ref_events))
    all_ref_pitches = np.concatenate((np.ones(len(BD_ref_events)), np.ones(len(SD_ref_events))*2, np.ones(len(HH_ref_events))*3))

    all_precision, all_recall, all_fmeasure = postProcessing.f_measure(all_est_events, all_ref_events, all_est_pitches, all_ref_pitches)
    global_results['precision'].append(all_precision)
    global_results['recall'].append(all_recall)
    global_results['fmeasure'].append(all_fmeasure)
    
#%% Visualization
i = 4 # n° test (see test_track_IDs)
print(dataset.data['audio_name'][test_track_IDs[i]])
print("Bass drum: precision = {0:.3f} ; recall = {0:.3f} ; fmeasure = {0:.3f}".format(BD_results['precision'][i] BD_results['recall'][i]; BD_results['fmeasure'][i]))
print("Snare drum: " + str(global_results['precision'][i]))
print("Hihat: " + str(global_results['precision'][i]))
print("Global: " + str(global_results['precision'][i]))

f, axes = plt.subplots(3, 1, sharex=True, sharey=True)

# BD
axes[0].plot(dataset.data['BD_target'][test_track_IDs[i]])
axes[0].plot(y_hat_grouped[i][:, 0])
axes[0].set_title('Activation function and ground-truth activation - Kick')
#print(postProcessing.activationToEvents(y_hat_grouped[i][:, 0], peak_thres = peak_thres))

# SD
axes[1].plot(dataset.data['SD_target'][test_track_IDs[i]])
axes[1].plot(y_hat_grouped[i][:, 1])
axes[1].set_title('Activation function and ground-truth activation - Snare')
#print(postProcessing.activationToEvents(y_hat_grouped[i][:, 1], peak_thres = peak_thres))

# HH
axes[2].plot(dataset.data['HH_target'][test_track_IDs[i]])
axes[2].plot(y_hat_grouped[i][:, 2])
axes[2].set_title('Activation function and ground-truth activation - Hihat')


#print(postProcessing.activationToEvents(y_hat_grouped[i][:, 2], peak_thres = peak_thres))
