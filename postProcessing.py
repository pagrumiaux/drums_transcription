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

def peakPickingThreshold(sequence, thres, m = 2, w = 7):
    peaks = []
    n_last_peak = -1-w
    length = sequence.shape[0]
    for n in range(length):
        if sequence[n] >= thres:
            if n < m:
                if sequence[n] == sequence[:n+m+1].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
#                    print("append")
            elif n+m > length:
                if sequence[n] == sequence[n-m:].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
#                    print("append")
            else:
                if sequence[n] == sequence[n-m:n+m+1].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
#                    print("append")
#        input("pause")
    return peaks

def peakVisualization(ground_truth_peaks, predicted_sequence):
    print('eok')
    
    
def activationToEvents(activation_function, peak_thres, sr = 44100, frame_rate = 100):
    peaks = peakPickingThreshold(activation_function, thres = peak_thres)
    peaks = np.array(peaks)
    events = peaks/frame_rate
    return events

def precisionRecallFmeasure(est_events, ref_events, est_pitches, ref_pitches, onset_tolerance = 0.02):
    ref_intervals = np.stack((np.array(ref_events), np.array(ref_events)+0.001)).T
    est_intervals = np.stack((np.array(est_events), np.array(est_events)+0.001)).T
    precision, recall, fmeasure, overlap = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=onset_tolerance, pitch_tolerance=0, offset_ratio=None)
    return precision, recall, fmeasure

def f_measure(est_events, ref_events, est_pitches, ref_pitches, onset_tolerance = 0.02):
    tp = 0
    for i, est in enumerate(est_events):
        for j, ref in enumerate(ref_events):
            if abs(est-ref) < onset_tolerance and est_pitches[i] == ref_pitches[j]:
                tp = tp + 1
    if est_events.shape[0] == 0 or ref_events.shape[0] == 0:
        if est_events.shape[0] == 0 and ref_events.shape[0] == 0:
            precision = 1.0
            recall = 1.0
        else:
            precision = 0.0
            recall = 0.0
    else:
        precision = tp/est_events.shape[0]
        recall = tp/ref_events.shape[0]
    if precision+recall == 0.0:
        fmeasure = 0.0
    else:
        fmeasure = 2*precision*recall/(precision+recall)
#    print(tp, est_events.shape[0], ref_events.shape[0])
    
    return precision, recall, fmeasure


def groupePredictionSamplesByTrack(y_test, test_IDs):
    y_test_grouped = []
    test_track_IDs = []
    n_dim = len(y_test.shape)
    
    cur_track_ID = test_IDs[0][0]
    list_to_concat = []
    for i, ID in enumerate(test_IDs):
        if n_dim == 3:
            if ID[0] != cur_track_ID:
                y_test_concat = np.concatenate(list_to_concat, axis=0)
                y_test_grouped.append(y_test_concat)
                list_to_concat = []
                test_track_IDs.append(cur_track_ID)
                cur_track_ID = ID[0]
                
    
            list_to_concat.append(y_test[i, :, :])   
            if ID == test_IDs[-1]:
                y_test_concat = np.concatenate(list_to_concat, axis=0)
                y_test_grouped.append(y_test_concat)
                test_track_IDs.append(cur_track_ID)
        
        elif n_dim == 2:
            if ID[0] != cur_track_ID:
                y_test_concat = np.stack(list_to_concat)
                y_test_grouped.append(y_test_concat)
                list_to_concat = []
                test_track_IDs.append(cur_track_ID)
                cur_track_ID = ID[0]
            list_to_concat.append(y_test[i, :])           

    return y_test_grouped, test_track_IDs

