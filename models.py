# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:18:19 2020

@author: PA
"""

import tensorflow as tf

def cnn_model(context_frames, n_bins):
    """Generate a tensorflow.keras.Model describing a CNN.

    Parameters
    ----------
    context_frames : int
        number of frames on each side of the analyzed frame, e.g. 
        context_frames=9 leads to a temporal dimension of size 2*9+1=19
        frames.
    n_bins : int
        number of frequency bins in the frequency dimension.

    Returns
    -------
    tensorflow.keras.Model
        Model containing all the layers (still need to be compiled with
        model.compile(), see training.py). The input is a magnitude 
        spectrogram of shape (2*context_frames+1, n_bins, 1)
    """
    # input layer
    input_shape = (context_frames, n_bins, 1)
    input0 = tf.keras.Input(shape=input_shape, name='input0')
    
    # convolutional block 1
    conv00 =  tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3), 
                                     activation='relu',
                                     padding='same', 
                                     kernel_initializer='he_uniform', 
                                     name='conv00'
                                     )(input0)
    conv01 =  tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3), 
                                     activation='relu',
                                     padding='same', 
                                     kernel_initializer='he_uniform', 
                                     name='conv01'
                                     )(conv00)
    norm01 = tf.keras.layers.BatchNormalization(name='norm01')(conv01)
    pool01 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name='pool01')(norm01)
    dropout01 = tf.keras.layers.Dropout(rate=0.5, name='dropout01')(pool01)
    
    # convolutional block 2
    conv02 =  tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3), 
                                     activation='relu',
                                     padding='same', 
                                     kernel_initializer='he_uniform', 
                                     name='conv02'
                                     )(dropout01)
    conv03 =  tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3), 
                                     activation='relu',
                                     padding='same', 
                                     kernel_initializer='he_uniform', 
                                     name='conv03'
                                     )(conv02)
    norm03 = tf.keras.layers.BatchNormalization(name='norm03')(conv03)
    pool03 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name='pool03')(norm03)
    dropout03 = tf.keras.layers.Dropout(rate=0.5, name='dropout03')(pool03)
    
    # dense block
    flatten00 = tf.keras.layers.Flatten(name='flatten00')(dropout03)
    dense00 = tf.keras.layers.Dense(units=256, activation='relu',
                                    kernel_initializer='he_uniform',
                                    name='dense00')(flatten00)
    dense01= tf.keras.layers.Dense(units=256, activation='relu',
                                    kernel_initializer='he_uniform',
                                    name='dense01')(dense00)
    
    # output layer
    output = tf.keras.layers.Dense(units=3, activation='sigmoid',
                                    kernel_initializer='he_uniform',
                                    name='output')(dense01)
    
    # model creation
    model = tf.keras.models.Model(inputs=input0, outputs=output)    
    return model

def rnn_model(n_frames, n_bins, units):
    """Generate a tensorflow.keras.Model describing a RNN.

    Parameters
    ----------
    n_frames : int
        number of frames for temporal analysis by the RNN.
    n_bins : int
        number of frequency bins in the frequency dimension.
    units : int
        number of units for the GRU layers.

    Returns
    -------
    tensorflow.keras.Model
        Model containing all the layers (still need to be compiled with
        model.compile(), see training.py). The input is a magnitude 
        spectrogram of shape (n_frames, n_bins)
    """
    # input layer
    input_shape = (n_frames, n_bins)
    input0 = tf.keras.Input(shape=input_shape, name='input0')
    
    # recurrent layers
    bgru00 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units, 
                                                               return_sequences=True), 
                                           name='bgru00')(input0)
    bgru01 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units, 
                                                               return_sequences=True), 
                                           name='bgru01')(bgru00)    
    # output
    output = tf.keras.layers.Dense(units=3, activation='sigmoid', name='output')(bgru01)
    
    # model creation
    model = tf.keras.models.Model(inputs=input0, outputs=output)
    return model

def crnn_model(n_frames, context_frames, n_bins, units):
    """Generate a tensorflow.keras.Model describing a CRNN.

    Parameters
    ----------
    n_frames : int
        number of spectrograms of size [2*context_frames+1, n_bins, 1] 
        for temporal analysis by the RNN part. Each spectrogram is 
        independently processed by the CNN part and the resulting outputs
        are putting in a temporal series as input into the RNN part.
    context_frames : int
        number of frames on each side of the analyzed frame, e.g. 
        context_frames=9 leads to a temporal dimension of size 2*9+1=19
        frames.
    n_bins : int
        number of frequency bins in the frequency dimension.
    units : int
        number of units for the GRU layers.

    Returns
    -------
    tensorflow.keras.Model
        Model containing all the layers (still need to be compiled with
        model.compile(), see training.py). The input of the model is a 
        series of magnitude spectrograms of shape (n_frames, context_frames n_bins, 1)
    """

    # CNN input layer
    cnn_input_shape = (context_frames, n_bins, 1)
    cnn_input = tf.keras.Input(shape=cnn_input_shape, name='cnn_input')
    
    # convolutional block 1
    conv00 = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv00'
                                     )(cnn_input)
    conv01 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='conv01'
                                    )(conv00)
    norm01 = tf.keras.layers.BatchNormalization(name='norm01')(conv01)
    pool01 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name='pool01')(norm01)
    dropout01 = tf.keras.layers.Dropout(rate=0.3, name='dropout01')(pool01)
    
    # convolutional block 2
    conv02 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv02'
                                     )(dropout01)
    conv03 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='conv03'
                                    )(conv02)
    norm03 = tf.keras.layers.BatchNormalization(name='norm03')(conv03)
    pool03 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), name='pool03')(norm03)
    dropout03 = tf.keras.layers.Dropout(rate=0.3, name='dropout03')(pool03)
    
    # CNN output
    cnn_output = tf.keras.layers.Flatten(name='flatten_cnn')(dropout03)
    
    # CNN model
    cnn_model = tf.keras.Model(inputs=cnn_input, outputs=cnn_output)
    
    # RNN input layer
    rnn_input_shape = (n_frames, context_frames, n_bins, 1)
    rnn_input = tf.keras.layers.Input(shape=rnn_input_shape)
    
    # TimeDistributed layer of the CNN model
    td00 = tf.keras.layers.TimeDistributed(cnn_model, name='td00')(rnn_input)
    
    # recurrent layers
    bgru00 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units, return_sequences=True), name='bgru00')(td00)
    bgru01 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units, return_sequences=True), name='bgru01')(bgru00)
    bgru02 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units, return_sequences=True), name='bgru02')(bgru01)
    
    # RNN output
    rnn_output = tf.keras.layers.Dense(units=3, activation='sigmoid', name='rnn_output')(bgru02)
    
    # crnn model
    crnn_model = tf.keras.Model(inputs=rnn_input, outputs=rnn_output)
    
    return crnn_model