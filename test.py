#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:27:50 2018

@author: grumiaux
"""

import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import keras.backend as K
import tensorflow as tf
import utilities

#%%
params = {'dim_x': 168,
          'dim_y': 25,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CNN',
          'context_frames': 25,
          'beats': False}

#%%
folder_rbma = './datasets/RBMA_13/'
folder_smt = './datasets/SMT_drums/'
folder_enst = None
enst_solo = False

dataset = Dataset(folder_rbma = folder_rbma, folder_smt = folder_smt, folder_enst = folder_enst)
dataset.load_dataset()

#%% three fold cross validation
list_IDs = dataset.generate_IDs(params['task'], dataFilter='smt')
n_ID = len(list_IDs)

training_IDs = list_IDs[163023:163023+params['batch_size']]
training_IDs_with_duplicate = utilities.duplicateTrainingSamples(dataset, training_IDs, ratio = 8)


#%%
for ID in training_IDs_with_duplicate:
    bd = dataset.data['BD_target'][ID[0]][ID[1]]
    sd = dataset.data['SD_target'][ID[0]][ID[1]]
    hh = dataset.data['HH_target'][ID[0]][ID[1]]
    print(bd, sd, hh)
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

#%%
#X = np.empty((params['batch_size'], params['dim_x'], params['dim_y'], 1))
#y = model.predict(X, verbose=1)
#print(y)

#%% generator
training_generator = DataGenerator(**params).generate(dataset, training_IDs)

#%% change learning rate
print(K.eval(model.optimizer.lr))
K.set_value(model.optimizer.lr, 0.00001)
print(K.eval(model.optimizer.lr))
#%% training
epochs = 5
print(K.eval(model.optimizer.lr))
for i in range(epochs):
    print('=== Epoch ' + str(i) + ' ===')
    model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], epochs=1)

#%% eval
X_test = np.empty((params['batch_size'], params['dim_x'], params['dim_y'], 1))

X, y = DataGenerator(**params).generateForTest(dataset, training_IDs)
X_test = X

y_hat = model.predict(X_test, verbose=1)
print(y_hat)

#%%
X, y = DataGenerator(**params).generateForTest(dataset, training_IDs)

#%% check gradients
outputTensor = model.output
trainable_weights = model.trainable_weights
gradients = K.gradients(outputTensor, trainable_weights)
trainingExample = np.random.random((1, 168, 25, 1))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
evaluated_gradients = sess.run(gradients, feed_dict={model.input:trainingExample})