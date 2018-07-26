#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:54 2018

@author: grumiaux
"""

#import keras
#from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GRU, Bidirectional, BatchNormalization, TimeDistributed
#from keras.models import Sequential, Model
#import keras.backend as K
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import utilities
import postProcessing

#%% Parameters
params = {'dim_x': 168,
          'dim_y': 9,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CBRNN',
          'context_frames': 9,
          'sequential_frames': 100,
          'beatsAndDownbeats': False}

dataFilter = "enst"

#%% Dataset load
dataset = Dataset()
dataset.loadDataset(enst_solo = False)

#%% all IDs
list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], sequential_frames = params['sequential_frames'], dataFilter=dataFilter)

# IDs three-fold repartition
train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_train_IDs']]
test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_test_IDs']]

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
optimizer = keras.optimizers.Adagrad()
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
peak_thres = 0.2
rec_half_length = 0

y_hat_grouped, test_track_IDs = postProcessing.groupePredictionSamplesByTrack(y_hat, list_IDs)
BD_results, SD_results, HH_results, beats_results, downbeats_results, global_results = postProcessing.computeResults(dataset, y_hat_grouped, test_track_IDs, peak_thres, rec_half_length, FmWithBeats = False)
print(np.mean(global_results['fmeasure']))

#%%
test_track_ID = 34 # n° test (see test_track_IDs)
postProcessing.visualizeModelPredictionPerTrack(test_track_ID, dataset, y_hat_grouped, test_track_IDs, BD_results, SD_results, HH_results, global_results)
