#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:49:55 2018

@author: grumiaux
"""

import pickle
import postProcessing
import os
import numpy as np

#%%
folder_soft = 'billboard/target/SMT-CNNb-spread0-Fm0.95/soft/'
folder_hard = 'billboard/target/SMT-CNNb-spread0-Fm0.95/hard/'

files = [f for f in os.listdir(folder_soft) if f.endswith('.npy')]
print(len(files))

#%%
for f in files:
    act = np.load(folder_soft + f)
    hard = postProcessing.activationToEvents(act, 0.2, rec_half_length = 0, sr = 44100, frame_rate = 100)
#    print(hard)
    with open(folder_hard + f[:-4], 'wb') as fi:
        pickle.dump(hard, fi)
        
    print(f + " done")