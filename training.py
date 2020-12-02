# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:51:01 2020

@author: PA
"""

from dataset import Dataset
# from dataGenerator import DataGenerator
import numpy as np
# import postProcessing

#%%
folder_rbma = "./datasets/RBMA_13/"
folder_smt = "./datasets/SMT_DRUMS/"
folder_enst = "./datasets/ENST_drums/"
folder_bb = "./datasets/billboard/"

#%% Parameters
params = {'dim_x': 168,
          'dim_y': 9,
          'batch_size': 8,
          'shuffle': True,
          'task': 'CBRNN',
          'context_frames': 9,
          'sequential_frames': 100,
          'beatsAndDownbeats': False}

dataFilter = "enst"

#%% Dataset load
dataset = Dataset(folder_rbma, folder_smt, folder_enst)
dataset.load_dataset(enst_solo = False)

#%%