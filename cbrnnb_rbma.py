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
import keras.backend as K
import manage_gpus as gpl
import time
import tensorflow as tf

# obtain lock
gpu_ids = gpl.board_ids()
if gpu_ids is None:
    gpu_device = None
else:
    gpu_device = -1

gpu_id_locked = -1
if gpu_device is not None:
    gpu_id_locked = gpl.obtain_lock_id(id=gpu_device)
    if gpu_id_locked < 0:
        time.sleep(3)
        if gpu_id_locked < 0:
            if gpu_device < 0:
                raise RuntimeError("No GPUs available for locking")
            else:
                raise RuntimeError("cannot obtain any of the selected GPUs {0}".format(str(gpu_device)))
    comp_device = "/GPU:0"
else:
    comp_device = "/cpu:0"

with tf.device(comp_device):
    # Parameters
    params = {'dim_x': 168,
              'dim_y': 13,
              'batch_size': 8,
              'shuffle': True,
              'task': 'CBRNN',
              'context_frames': 13,
              'beatsAndDownbeats': False, 
              'multiTask': False,
              'difference_spectrogram': True,
              'sequential_frames': 400}
    
    dataFilter = 'rbma'
    
    # Dataset load
    dataset = Dataset()
    dataset.loadDataset(enst_solo = False)
       
    # all IDs
    list_IDs = dataset.generate_IDs(params['task'], context_frames = params['context_frames'], sequential_frames = params['sequential_frames'], dataFilter=dataFilter)
    
    # IDs three-fold repartition
    train_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_train_IDs']]
    test_IDs = [ID for ID in list_IDs if ID[0] in dataset.split[dataFilter + '_test_IDs']]
    
    n_train_IDs = int(0.85*len(train_IDs))
    training_IDs = train_IDs[:n_train_IDs]
    validation_IDs = train_IDs[n_train_IDs:]
    
    # Model creation
    
    cnn_input_shape = (params['dim_x'], params['dim_y'], 1)
#    units = 30
    units = 60
    
    ### cnn layer
    cnn_input = Input(shape=cnn_input_shape, name = 'cnn_input')
    
    # CNN block A
    x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(cnn_input)
    x_cnn = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
#    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
#   x_cnn = Dropout(0.3)(x_cnn)
    
    # CNN block B
    x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
    x_cnn = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x_cnn)
#    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D(pool_size=(3, 3))(x_cnn)
#    x_cnn = Dropout(0.3)(x_cnn)
    cnn_output = Flatten()(x_cnn)
    
    cnn_model = Model(inputs=cnn_input, outputs = cnn_output) # cnn model which will treat each timestep before the rnn
    
    # RNN
    rnn_input_shape = (params['sequential_frames'], params['dim_x'], params['dim_y'], 1)
    rnn_input = Input(shape=rnn_input_shape)
    
    x_rnn = TimeDistributed(cnn_model)(rnn_input)
    
    x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
    x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
    x_rnn = Bidirectional(GRU(units, return_sequences=True))(x_rnn)
    if not params['multiTask']:
        rnn_output = Dense(3, activation='sigmoid')(x_rnn)
    else:
        rnn_output = Dense(5, activation='sigmoid')(x_rnn)
    
    cbrnn_model = Model(inputs=rnn_input, outputs=rnn_output)
    optimizer = keras.optimizers.RMSprop(lr=0.0005)
    cbrnn_model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    #
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
    
        cbrnn_model.fit_generator(generator = training_generator, steps_per_epoch = len(training_IDs)//params['batch_size'], validation_data = validation_generator, validation_steps = len(validation_IDs)//params['batch_size'], epochs=1)
    
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
                if refinement:
                    K.set_value(optimizer.lr, cur_lr/3.)
                no_improv_count = 0
        
        print("---> val loss: " + str(cur_val_loss) + " ; val acc: " + str(cur_val_acc))
    cbrnn_model.save('RBMA-CBRNNb-nodropout-epochs{}-vacc{}.hdf5'.format(i, cur_val_acc))


if gpu_id_locked >= 0:
    gpl.free_lock(gpu_id_locked)
