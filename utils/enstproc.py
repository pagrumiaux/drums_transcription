#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:39:22 2018

@author: grumiaux
"""

from shutil import copy2
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
#%%
acc_folder = "./ENST-drums/audio/accompaniment/"
drums_folder = "./ENST-drums/audio/drums/"
mix_folder = "./ENST-drums/audio/mix66/"
acc = os.listdir(acc_folder)
drums = os.listdir(drums_folder)

#for i, e in enumerate(acc):
#    if e != drums[i]:
#        print(i)
        
#%%
for i in range(len(acc)):
    fs_acc, x_acc = scipy.io.wavfile.read(acc_folder + acc[i])
    fs_drum, x_drum = scipy.io.wavfile.read(drums_folder + drums[i])
    
    if fs_acc == fs_drum:
        y = 0.33*x_acc + 0.66*x_drum
        y = y / np.max(y)
#        print(np.max(y))
            
        scipy.io.wavfile.write(mix_folder + acc[i], fs_acc, y)
        print(str(i) + " === " + acc[i] + " done ===")
    
    else:
        print(i)
        
        
#%%
annot_folder = "./ENST-drums/annotations/d3/"
for e in os.listdir(annot_folder):
    copy2(annot_folder + e, annot_folder + "drummer_3_" + e)
    print(e)