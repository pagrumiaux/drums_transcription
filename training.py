# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:51:01 2020

@author: PA
"""

from dataset import Dataset
from data_generator import DataGenerator
import models

import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime
# import postProcessing

#%% tf settings
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

#%%
folder_rbma = "./datasets/RBMA_13/"
# folder_rbma = None
# folder_smt = "./datasets/SMT_DRUMS/"
folder_smt = None
# folder_enst = "./datasets/ENST_drums/"
folder_enst = None
# folder_bb = "./datasets/billboard/"


#%% Training parameters
params_train = {'batch_size': 64,
          'model': 'RNN',
          'n_bins': 168,
          'sequential_frames': 100,          
          'context_frames': 9,
          'shuffle': True,
          'difference_spectrogram': True,
          'beats_and_downbeats': False
          }

#%% Dataset load
dataset = Dataset(folder_rbma=folder_rbma, folder_smt=folder_smt, folder_enst=folder_enst, 
                  enst_solo = False)

print('All datasets loaded.')

#%% generate the IDs for the data generator (train, valid and test)
list_IDs = dataset.generate_IDs(params_train['model'], 
                                context_frames = params_train['context_frames'], 
                                sequential_frames = params_train['sequential_frames'])


train_valid_IDs = [idx for idx in list_IDs if 
             dataset.data['audio_name'][idx[0]] in dataset.split['rbma_train_files']
             or dataset.data['audio_name'][idx[0]] in dataset.split['smt_train_files']
             or dataset.data['audio_name'][idx[0]] in dataset.split['enst_train_files']
             ]

test_IDs = [idx for idx in list_IDs if 
             dataset.data['audio_name'][idx[0]] in dataset.split['rbma_test_files']
             or dataset.data['audio_name'][idx[0]] in dataset.split['smt_test_files']
             or dataset.data['audio_name'][idx[0]] in dataset.split['enst_test_files']
             ]

# 15% of training IDs is used for validation
np.random.shuffle(train_valid_IDs)
n_train_IDs = int(0.85*len(train_valid_IDs))
valid_IDs = train_valid_IDs[n_train_IDs:]
train_IDs = train_valid_IDs[:n_train_IDs]

print('Training, validation and test IDs created')

#%% instantiating the training and validation generators
params_train['dataset'] = dataset
params_valid = copy.deepcopy(params_train)

params_train['list_IDs'] = train_IDs
params_valid['list_IDs'] = valid_IDs

train_generator = DataGenerator(**params_train)
valid_generator = DataGenerator(**params_valid)

print("Training and validation generators instantiated.")

#%% model creation
model_name = "rnn_100frames_RBMA"

# parameters
rnn_units = 30

if params_train['model'] == 'CNN':
    model = models.cnn_model(context_frames=params_train['context_frames'], n_bins=params_train['n_bins'])
elif params_train['model'] == 'RNN':
    model = models.rnn_model(n_frames=params_train['sequential_frames'], n_bins=params_train['n_bins'], units=rnn_units)
elif params_train['model'] == 'CBRNN':
    model = models.cbrnn_model(n_frames=params_train['sequential_frames'], context_frames=params_train['context_frames'], n_bins=params_train['n_bins'], units=rnn_units)

adam = tf.keras.optimizers.Adam()
model.compile(optimizer=adam,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['categorical_accuracy'])

date = datetime.datetime.now().strftime("%Y-%m-%d")
model_path = "models/" + date + "_" + model_name

print(f'Model will be saved in {model_path}')
print('Model architecture :')
print(model.summary())

#%% training
epochs = 300

earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy',
                                             patience = 50
                                            )
saveBest = tf.keras.callbacks.ModelCheckpoint(model_path, 
                                              monitor = 'val_categorical_accuracy', 
                                              save_best_only = True,
                                              verbose = 1
                                             )
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                                                factor = 0.33,
                                                patience = 10,
                                                verbose = 1
                                               )

tic = time.time()
model.fit_generator(generator = train_generator,
                    validation_data = valid_generator,
                    epochs = epochs,
                    verbose = 1,
                    callbacks = [earlyStop, saveBest, reduceLR],
                   )

toc = time.time()
training_time = datetime.timedelta(seconds=toc-tic)
print(f'\nTime spent for training : {training_time}')