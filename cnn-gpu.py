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
          'dim_y': 25,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 25,
          'beatsAndDownbeats': False, 
          'multiTask': False,
          'difference_spectrogram': True}

index_first_solo_drums = 131

#%% Dataset load
dataset = Dataset()
dataset.loadDataset(spread_length = 5)

#%% all IDs
list_IDs = dataset.generate_IDs(params['task'], stride = 0, context_frames = params['context_frames'], dataFilter='rbma')

# IDs three-fold repartition
train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split['rbma_train_IDs']]
test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split['rbma_test_IDs']]

n_train_IDs = int(0.85*len(train_IDs))
training_IDs = train_IDs[:n_train_IDs]
validation_IDs = train_IDs[n_train_IDs:]
   
#%%
### model creation ###
input_shape = (params['dim_x'], params['context_frames'], 1)

model = Sequential()
# block A
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_initializer='he_uniform'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(1.0))
# block B
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(1.0))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='sigmoid', kernel_initializer='he_uniform'))

optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

LRPlateau = ReduceLROnPlateau(factor=0.5, verbose=1, patience=10)
Checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

#%%
patience = 5
epochs = 50
refinement = False

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
            if refinement:
                K.set_value(optimizer.lr, cur_lr/3.)
            no_improv_count = 0
    
    print("---> val loss: " + str(cur_val_loss) + " ; val acc: " + str(cur_val_acc))
    model.save('RBMA-CNNb-epochs{}-vacc{}.hdf5'.format(i, cur_val_acc))
#%%
#X_test = np.empty((len(test_IDs), params['dim_x'], params['dim_y'], 1))
#for i in range(len(test_IDs)):
#    X_test[i, :, :, 0] = DataGenerator(**params).extract_feature(dataset, test_IDs[i])
#    
#y_hat = model.predict(X_test, verbose=1)
