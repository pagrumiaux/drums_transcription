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
import numpy as np

#%%
params = {'dim_x': 100,
          'dim_y': 84,
          'batch_size': 8,
          'shuffle': True,
          'task': 'RNN',
          'sequential_frames': 100}
np.random.seed(1)

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
limit_rbma = 27
list_IDs = dataset.generate_IDs(params['task'], context_frames=params['sequential_frames'], dataFilter='smt')
n_ID = len(list_IDs)
split_train_test = int(n_ID*0.67)
train_index = int(0.85*split_train_test)

test_IDs = list_IDs[split_train_test:]
temp_IDs = list_IDs[:split_train_test]
np.random.shuffle(temp_IDs)
training_IDs = temp_IDs[:train_index]
validation_IDs = temp_IDs[train_index:]

#%%
# Generators
training_generator = DataGenerator(**params).generate(dataset, training_IDs)
validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

#%%
units = 50
input_shape = (params['dim_x'], params['dim_y'], 1)

model = Sequential()
model.add(Bidirectional(GRU(units, return_sequences=True), input_shape=(100, 84)))
model.add(Bidirectional(GRU(units, return_sequences=True)))
model.add(Dense(2, activation='sigmoid'))
rms = RMSprop(lr=0.007)
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['acc'])
#print(model.summary())
#%%
model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1)