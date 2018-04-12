#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:26:06 2018

@author: grumiaux
"""

def duplicateTrainingSamples(dataset, training_IDs, ratio = 10):
    training_IDs_with_duplicate = []
    for ID in training_IDs:
        training_IDs_with_duplicate.append(ID)
        bd = dataset.data['BD_target'][ID[0]][ID[1]]
        sd = dataset.data['SD_target'][ID[0]][ID[1]]
        hh = dataset.data['HH_target'][ID[0]][ID[1]]
        if bd or sd or hh:

            for i in range(ratio-1):
                training_IDs_with_duplicate.append(ID)
    return training_IDs_with_duplicate


def spreadTargetFunctions(target_function, spread_length):
    new_target_function = target_function.copy()
    target_length = len(new_target_function)
    countdown = 0
    for i in range(target_length):
        if new_target_function[i] == 1.0:
            countdown = min(spread_length, target_length-i)
            for j in range(min(spread_length, i)):
                new_target_function[i-(j+1)] = 1.0
        elif countdown >= 0:  
            new_target_function[i] = 1.0
            countdown = countdown - 1
    
    return new_target_function