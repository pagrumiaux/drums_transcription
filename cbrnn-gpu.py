#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:54 2018

@author: grumiaux
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GRU, Bidirectional, BatchNormalization, TimeDistributed
from keras.models import Sequential, Model
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import utilities
import postProcessing
import matplotlib.pyplot as plt
import keras.backend as K

#%% Parameters
params = {'dim_x': 168,
          'dim_y': 13,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CBRNN',
          'context_frames': 13,
          'sequential_frames': 400}

index_first_solo_drums = 131

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], dataFilter='smt')
n_ID = len(list_IDs)

#%% all IDs
list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], dataFilter='smt')

# IDs three-fold repartition
train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split['smt_train_IDs']]
test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split['smt_test_IDs']]

n_train_IDs = int(0.85*len(train_IDs))
training_IDs = train_IDs[:n_train_IDs]
validation_IDs = train_IDs[n_train_IDs:]

#%% Model creation

cnn_input_shape = (params['dim_x'], params['dim_y'], 1)
#units = 30
units = 60

### cnn layer
cnn_input = Input(shape=cnn_input_shape, name = 'cnn_input')

# CNN block A
x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(cnn_input)
x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
x_cnn = BatchNormalization()(x_cnn)
x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
x_cnn = Dropout(0.3)(x_cnn)

# CNN block B
x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
x_cnn = BatchNormalization()(x_cnn)
x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
x_cnn = Dropout(0.3)(x_cnn)
cnn_output = Flatten()(x_cnn)

cnn_model = Model(inputs=cnn_input, outputs = cnn_output) # cnn model which will treat each timestep before the rnn

# RNN
rnn_input_shape = (params['sequential_frames'], params['dim_x'], params['dim_y'], 1)
rnn_input = Input(shape=rnn_input_shape)

x_rnn = TimeDistributed(cnn_model)(rnn_input)

x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
rnn_output = Dense(3, activation='sigmoid')(x_rnn)

cbrnn_model = Model(inputs=rnn_input, outputs=rnn_output)
print(cbrnn_model.summary())
optimizer = keras.optimizers.RMSprop(lr=0.0005)
cbrnn_model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#%%
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

    np.random.shuffle(training_IDs)
       
    training_IDs_with_duplicate = utilities.duplicateTrainingSamples(dataset, training_IDs, ratio = 12)
    training_generator = DataGenerator(**params).generate(dataset, training_IDs_with_duplicate)    
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

    cbrnn_model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs_with_duplicate)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1)

    # check validation accuracy evolution for refinement
    cur_val_loss, cur_val_acc = cbrnn_model.evaluate_generator(validation_generator, steps=len(validation_IDs)//params['batch_size'])
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
X_test = np.empty((len(test_IDs), params['sequential_frames'], params['dim_x'], params['dim_y'], 1))
for i in range(len(test_IDs)):
    X_test[i, :, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])

y_hat = cbrnn_model.predict(X_test, verbose=1)
#%%
y_hat = np.empty((len(test_IDs), params['sequential_frames'], 3))
for i in range(len(test_IDs)):
    print(str(i) + "/" + str(len(test_IDs)))
    y_hat[i, :, :] = cbrnn_model.predict(X_test[i, :, :, :, :])

#%% F measure for BD, SD and HH on the test set
peak_thres = 0.2

#BD_fmeasure = []
#SD_fmeasure = []
#HH_fmeasure = []
global_fmeasure = []
y_hat_grouped, test_track_IDs = postProcessing.groupePredictionSamplesByTrack(y_hat, test_IDs)

for i, ID in enumerate(test_track_IDs):
#    print(i, ID)
#    i = i+7
    # BD events
    BD_est_activation = y_hat_grouped[i][:, 0]
    BD_est_events = postProcessing.activationToEvents(BD_est_activation, peak_thres = peak_thres)
    BD_ref_events = dataset.data['BD_annotations'][ID]
#    print(BD_ref_events, len(BD_ref_events))
#    print(BD_est_events, len(BD_est_events))
#    input('pause')
    
    # SD events
    SD_est_activation = y_hat_grouped[i][:, 1]
    SD_est_events = postProcessing.activationToEvents(SD_est_activation, peak_thres = peak_thres)
    SD_ref_events = dataset.data['SD_annotations'][ID]
#    print(SD_ref_events)
    
    # HH events
    HH_est_activation = y_hat_grouped[i][:, 2]
    HH_est_events = postProcessing.activationToEvents(HH_est_activation, peak_thres = peak_thres)
    HH_ref_events = dataset.data['HH_annotations'][ID]
#    print(HH_est_activation)
#    print(HH_est_events)
#    print(HH_ref_events)
    
    # all events
    est_events = np.concatenate((BD_est_events, SD_est_events, HH_est_events))
    est_pitches = np.concatenate((np.ones(len(BD_est_events)), np.ones(len(SD_est_events))*2, np.ones(len(HH_est_events))*3))
    ref_events = np.concatenate((BD_ref_events, SD_ref_events, HH_ref_events))
    ref_pitches = np.concatenate((np.ones(len(BD_ref_events)), np.ones(len(SD_ref_events))*2, np.ones(len(HH_ref_events))*3))

#    print(len(est_events), len(ref_events))
#    _, _, fmeasure = postProcessing.precisionRecallFmeasure(est_events, ref_events, est_pitches, ref_pitches)
    precision, recall, fmeasure = postProcessing.f_measure(est_events, ref_events, est_pitches, ref_pitches)
#    print(precision, recall, fmeasure)
#    input("pause")
    global_fmeasure.append(fmeasure)
    
#%% Visualization
i = 40 # n° test (see test_track_IDs)
print(dataset.data['audio_name'][test_track_IDs[i]], global_fmeasure[i])

# BD
plt.figure(1)
plt.title('Activation function and ground-truth activation - Kick')
plt.ylabel('Activation')
plt.xlabel('Frames')
plt.plot(dataset.data['BD_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 0])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 0], peak_thres = peak_thres))
# SD
plt.figure(2)
plt.title('Activation function and ground-truth activation - Snare')
plt.ylabel('Activation')
plt.xlabel('Frames')
plt.plot(dataset.data['SD_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 1])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 1], peak_thres = peak_thres))
# HH
plt.figure(3)
plt.title('Activation function and ground-truth activation - Hihat')
plt.ylabel('Activation')
plt.xlabel('Frames')
plt.plot(dataset.data['HH_target'][test_track_IDs[i]])
plt.plot(y_hat_grouped[i][:, 2])
print(postProcessing.activationToEvents(y_hat_grouped[i][:, 2], peak_thres = peak_thres))
