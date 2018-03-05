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
import numpy as np
import keras.backend as K

#%%
params = {'dim_x': 84,
          'dim_y': 25,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN'}


#%% Dataset load
dataset = Dataset()
dataset.loadDataset()

# %%three fold cross validation
list_IDs = dataset.generate_IDs(params['task'])
n_ID = len(list_IDs)
split_train_test = 535698
train_index = int(0.85*split_train_test)

test_IDs = list_IDs[split_train_test:]
temp_IDs = list_IDs[:split_train_test]

#%%
# Generators
training_generator = DataGenerator(**params).generate(dataset, training_IDs)
validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)

#%%
### model creation ###
input_shape = (84, 25, 1)
epochs = 5

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

#%%
for i in range(epochs):
    np.random.seed(i)
    np.random.shuffle(temp_IDs)
    training_IDs = temp_IDs[:train_index]
    validation_IDs = temp_IDs[train_index:]
    training_generator = DataGenerator(**params).generate(dataset, training_IDs)    
    validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)
    
    print(K.get_value(optimizer.lr))
    model.fit_generator(generator = training_generator, steps_per_epoch = len(list_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1)
#    K.set_value(optimizer.lr, 0.0005)
#%%
X_test = np.empty((len(test_IDs), 84, 25, 1))
for i, ID in enumerate(test_IDs):
    X_test[i, :, :, 0] = DataGenerator(84, 25, 'CNN').extract_feature(dataset, ID)
    
y_test = model.predict(X_test)