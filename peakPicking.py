#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:40:20 2018

@author: grumiaux
"""
import numpy as np
import matplotlib.pyplot as plt
import mir_eval

def peakPicking(sequence, m = 2, a = 2, w = 5, thres = 0.1):
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
#                    print("peak")
                    peaks.append(n)
                    n_last_peak = n
    
    return peaks

def peakVisualization(ground_truth_peaks, predicted_sequence):
    print('eok')
    
    
def activationToEvents(activation_function, peak_thres, sr = 44100, frame_rate = 100):
    peaks = peakPicking(activation_function, thres = peak_thres)
    peaks = np.array(peaks)
    events = peaks/frame_rate
    return events

def precisionRecallFmeasure(est_events, ref_events, onset_tolerance = 0.02):
    ref_intervals = np.stack((np.array(ref_events), np.array(ref_events)+0.001)).T
    est_intervals = np.stack((np.array(est_events), np.array(est_events)+0.001)).T
    precision, recall, fmeasure = mir_eval.transcription.onset_precision_recall_f1(ref_intervals, est_intervals, onset_tolerance=onset_tolerance)
    return precision, recall, fmeasure