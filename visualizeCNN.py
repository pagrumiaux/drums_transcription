#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:46:24 2018

@author: grumiaux
"""
import keras
from keras import backend as K
import numpy as np
from imageio import imwrite
#%% images dimensions
img_width = 25
img_height = 168
#%% convert tensor into image
def deprocess_image(x):
    #normalize
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1
    
    #clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    #convert ro RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%% model load
model = keras.models.load_model('test_CNN.hdf5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = model.input

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

#%%
layer_name = 'dense_3'

filters = []
for filter_index in range(1):
    print('Processing filter %d' % filter_index)

    # loss function to maximize the activation
    layer_output = layer_dict[layer_name].output
#    if K.image_data_format() == 'channel_first':
#        loss = K.mean(layer_output[:, filter_index, :, :])
#    else:
#        loss = K.mean(layer_output[:, :, :, filter_index])
        
    loss = K.mean(layer_output[:, 3])
        
    # compute the gradient
    grads = K.gradients(loss, input_img)[0]
    
    # normalize the gradient
    grads = normalize(grads)
    print(grads.shape)
    
    # returns the loss and grads
    iterate = K.function([input_img], [loss, grads])
    
    #step size for the gradient ascent
    step = 1

    # we start from a gray imge with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_height, img_width))
    else:
        input_img_data = np.random.random((1, img_height, img_width, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    
    # gradient ascent
    for i in range(200):
        loss_value, grads_value = iterate([input_img_data])
        print(loss_value)
        input_img_data += grads_value * step
        
        print('Current loss value:', loss_value)
        if loss_value < 0.:
            img = deprocess_image(input_img_data[0])
            filters.append((img, loss_value))
            break
        
        if loss_value >= 0:
            img = deprocess_image(input_img_data[0])
            filters.append((img, loss_value))
    print('Ok')
        
#%%
n = 1

# filters with highest loss are assumed to be better looking
filters.sort(key=lambda x: x[1], reverse=True)
filters = filters[:n*n]

margin = 5
width = n*img_width + (n-1)*margin
height = n*img_height + (n-1)*margin
stitched_filters = np.zeros((height, width, 1))

for i in range(n):
    for j in range(n):
        img, loss = filters[i*n+j]
        stitched_filters[(img_height + margin) * j: (img_height + margin) * j + img_height, (img_width+margin) * i: (img_width+margin) * i + img_width,  :] = img

#%%
imwrite('hihat.png', img)
    