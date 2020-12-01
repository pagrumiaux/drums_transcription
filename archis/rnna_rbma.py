#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:54 2018

@author: grumiaux
"""

import keras
from keras.layers import GRU, Bidirectional, Dense
from keras.models import Sequential
from dataset import Dataset
from dataGenerator import DataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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
    params = {'dim_x': 100,
              'dim_y': 168,
              'batch_size': 8,
              'shuffle': True,
              'task': 'RNN',
              'sequential_frames': 100,
              'beatsAndDownbeats': False, 
              'multiTask': False,
              'difference_spectrogram': True}
    
    dataFilter = 'rbma'
    
    # Dataset load
    dataset = Dataset()
    dataset.loadDataset(enst_solo = False)
    
    # all IDs
    list_IDs = dataset.generate_IDs(params['task'], stride = 0, sequential_frames = params['sequential_frames'], dataFilter=dataFilter)
    
    # IDs three-fold repartition
    train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_train_IDs']]
    test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter+'_test_IDs']]
    
    n_train_IDs = int(0.85*len(train_IDs))
    training_IDs = train_IDs[:n_train_IDs]
    validation_IDs = train_IDs[n_train_IDs:]

    ### model creation ###
    units = 50
    input_shape = (params['sequential_frames'], params['dim_y'])
    
    model = Sequential()
    model.add(Bidirectional(GRU(units, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(GRU(units, return_sequences=True)))
    #model.add(Bidirectional(GRU(units, return_sequences=True)))
    if not params['multiTask']:
        model.add(Dense(3, activation='sigmoid'))
    else:
        model.add(Dense(5, activation='sigmoid'))
    
    optimizer = keras.optimizers.RMSprop(lr=0.007)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    LRPlateau = ReduceLROnPlateau(factor=0.5, verbose=1, patience=10)
    Checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
    
    
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
    
        model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1, callbacks=[LRPlateau])
    
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
    model.save('RBMA-RNNa-epochs{}-vacc{}.hdf5'.format(i, cur_val_acc))


if gpu_id_locked >= 0:
    gpl.free_lock(gpu_id_locked)
