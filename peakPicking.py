#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:40:20 2018

@author: grumiaux
"""
import numpy as np
import matplotlib.pyplot as plt

def peakPicking(sequence, m = 2, a = 2, w = 2, thres = 0.15):
    """ Detects peaks in a sequence
        input:
            numpy array of the sequence data
    """
    peaks = []
    n_last_peak = 0
    length = sequence.shape[0]
    for n in range(max(a, m), length):
        if sequence[n] == np.max(sequence[n-m:n+1]):
            if sequence[n] >= (np.mean(sequence[n-a:n+1])+thres):
                if (n-n_last_peak) > w:
                    print("peak")
                    peaks.append(n)
                    n_last_peak = n
    
    return peaks

def peakVisualization(ground_truth_peaks, predicted_sequence):
    print('eok')