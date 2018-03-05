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

#%% Parameters
params = {'dim_x': 84,
          'dim_y': 25,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 25,
          'sequential_frames': 100}
np.random.seed(1)

#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'])
n_ID = len(list_IDs)
split_train_test = 535698
train_index = int(0.85*split_train_test)

#%%
test_IDs = list_IDs[split_train_test:]
temp_IDs = list_IDs[:split_train_test]
np.random.shuffle(temp_IDs)
training_IDs = temp_IDs[:train_index]
validation_IDs = temp_IDs[train_index:]

#%% Generators
training_generator = DataGenerator(**params).generate(dataset, training_IDs)
validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

#%% Model creation

cnn_input_shape = (params['dim_x'], params['dim_y'], 1)
units = 60

### cnn layer
cnn_input = Input(shape=cnn_input_shape, name = 'cnn_input')

# CNN block A
x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu')(cnn_input)
x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu')(x_cnn)
x_cnn = BatchNormalization()(x_cnn)
x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
x_cnn = Dropout(0.3)(x_cnn)

# CNN block B
x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu')(x_cnn)
x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu')(x_cnn)
x_cnn = BatchNormalization()(x_cnn)
x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
x_cnn = Dropout(0.3)(x_cnn)
cnn_output = Flatten()(x_cnn)

cnn_model = Model(inputs=cnn_input, outputs = cnn_output) # cnn model which will treat each timestep before the rnn

# RNN
rnn_input_shape = (params['sequential_frames'], params['dim_x'], params['dim_y'], 1)
rnn_input = Input(shape=rnn_input_shape)

x_rnn = TimeDistributed(cnn_model)(rnn_input)
cbrnn_model = Model(inputs=rnn_input, outputs=x_rnn)

x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
rnn_output = Dense(2, activation='sigmoid')(x_rnn)

cbrnn_model = Model(inputs=rnn_input, outputs=rnn_output)
print(cbrnn_model.summary())

#%% Model training
cbrnn_model.fit_generator(generator = training_generator, steps_per_epoch = len(list_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'],epochs=1)

