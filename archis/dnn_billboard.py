#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:54 2018

@author: grumiaux
"""

from keras import optimizers
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from dataset import Dataset
from dataGenerator import DataGenerator
import numpy as np
import keras.backend as K
import manage_gpus as gpl
import time
import tensorflow as tf

#%% obtain lock
# gpu_ids returns the list of gpus available for locking on the current systeme 
# the result will be None on systems without gpu nvidia card
gpu_ids = gpl.board_ids()

if gpu_ids is None:
     # system does not have a GPU so don't bother locking one, directly use CPU
     gpu_device=None
else:
     # select any free gpu
     gpu_device=-1

gpu_id_locked = -1
if gpu_device is not None:
    gpu_id_locked = gpl.obtain_lock_id(id=gpu_device)
    if gpu_id_locked < 0:
        # automatic lock removal has time delay of 2 so be sure to have the lock of the last run removed we wait
        # for 3 s here
        time.sleep(3)
        gpu_id_locked=gpl.obtain_lock_id(id=gpu_device)
        if gpu_id_locked < 0:
            if gpu_device < 0:
                raise RuntimeError("No GPUs available for locking")
            else:
                raise RuntimeError("cannot obtain any of the selected GPUs {0}".format(str(gpu_device)))

    # obtain_lock_id positions CUDA_VISIBLE_DEVICES such that only the selected GPU is visibale,
    # therefore we need now select /GPU:0
    comp_device = "/GPU:0"
else:
    comp_device = "/cpu:0" 


with tf.device(comp_device):
    params = {'dim_x': 168,
              'dim_y': 9,
              'batch_size': 8,
              'shuffle': True,
              'task': 'DNN',
              'context_frames': 9,
              'beatsAndDownbeats': False, 
              'multiTask': False,
              'difference_spectrogram': True}
    
    dataFilter = "enst"
    
    # Dataset load
    dataset = Dataset()
    dataset.loadDataset(enst_solo = False)
    
    # all IDs
    list_IDs = dataset.generate_IDs(params['task'], stride = 0, context_frames = params['context_frames'], dataFilter=dataFilter)
    
    # IDs three-fold repartition
    train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_train_IDs']]
    test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_test_IDs']]
    
    n_train_IDs = int(0.85*len(train_IDs))
    training_IDs = train_IDs[:n_train_IDs]
    validation_IDs = train_IDs[n_train_IDs:]

    input_shape = params['dim_x']

    model = Sequential()
    model.add(Dense(168, input_shape=(784,)))
    model.add(Dense(168, input_shape=(784,), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    if not params['multiTask']:
        model.add(Dense(3, activation='sigmoid'))
    else:
        model.add(Dense(5, activation='sigmoid'))
    
    optimizer = optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # epochs loop
    patience = 5
    epochs = 30
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
           
#        training_IDs_with_duplicate = utilities.duplicateTrainingSamples(dataset, training_IDs, ratio = 12)
        training_generator = DataGenerator(**params).generate(dataset, training_IDs)    
        validation_generator = DataGenerator(**params).generate(dataset, validation_IDs)
    
        model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1)
    
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
    model.save('ENSTsolo-CNNa-dropout0.3-epochs{}-vacc{}.hdf5'.format(i, cur_val_acc))


if gpu_id_locked >= 0:
    gpl.free_lock(gpu_id_locked)
