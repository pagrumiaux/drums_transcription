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

#%%
params = {'dim_x': 168,
          'dim_y': 25,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN'}

index_first_solo_drums = 131

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], dataFilter='smt')
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

temp_IDs = list_IDs[ID_first_solo:] #temporary list of IDs (will be for training and validation). We start putting solo drums for training in this list
test_IDs = [] #empty list of IDs for evaluation

while len(temp_IDs) < int(2*n_ID/3): # two third of the IDs will be for training and validation
    temp_IDs = temp_IDs + [ID for ID in list_IDs if ID[0] == list_mix_first_IDs[-1]] #we select all IDs of one song each loop
    list_mix_first_IDs = list_mix_first_IDs[:-1] # we remove this ID[0] of the list
    
for first_ID in list_mix_first_IDs: # we put the rest of IDs in the evaluation IDs list
    test_IDs = test_IDs + [ID for ID in list_IDs if ID[0] == first_ID]

train_valid_split = int(0.85*len(temp_IDs)) # we divide the temporary list into training and validation (85% for training)
training_IDs = temp_IDs[:train_valid_split]
validation_IDs = temp_IDs[train_valid_split:]

##%%
## Generators
#training_generator = DataGenerator(**params).generate(dataset, training_IDs)
#validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

#%%
### model creation ###
input_shape = (168, 25, 1)
epochs = 10

model = Sequential()
# block A
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))
# block B
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
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
for i in range(epochs):
    print("=== Epoch n°" + str(i))
    np.random.seed(i)
    np.random.shuffle(temp_IDs)
    training_IDs = temp_IDs[:train_valid_split]
    validation_IDs = temp_IDs[train_valid_split:]
    training_generator = DataGenerator(**params).generate(dataset, training_IDs)    
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)
    
    print(K.get_value(optimizer.lr))
    model.fit_generator(generator = training_generator, steps_per_epoch = len(list_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1, callbacks=[LRPlateau])
#%%
X_test = np.empty((len(test_IDs), 84, 25, 1))
for i, ID in enumerate(test_IDs):
    X_test[i, :, :, 0] = DataGenerator(84, 25, 'CNN').extract_feature(dataset, ID)
    
y_test = model.predict(X_test)