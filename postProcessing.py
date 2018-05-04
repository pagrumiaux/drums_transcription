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
                    peaks.append(n)
                    n_last_peak = n
    
    return peaks

def peakPickingThreshold(sequence, thres, m = 2, w = 10):
    peaks = []
    n_last_peak = -1-w
    length = sequence.shape[0]
    for n in range(length):
        if sequence[n] >= thres:
            if n < m:
                if sequence[n] == sequence[:n+m+1].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
            elif n+m > length:
                if sequence[n] == sequence[n-m:].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
            else:
                if sequence[n] == sequence[n-m:n+m+1].max() and n-n_last_peak > w:
                    peaks.append(n)
                    n_last_peak = n
    return peaks

def peakPickingRectangular(sequence, thres, w):
    return None

def peakPickingRectangularByMean(sequence, thres, rec_half_length = 0, w = 10):
    peaks = []
    n_last_peak = -1-w
    length = sequence.shape[0]
    padding = rec_half_length*2
    pad_sequence = np.concatenate((np.zeros((padding)), sequence, np.zeros((padding))))
    for n in range(length):
        if pad_sequence[padding+n] >= thres:
            n_mean = np.mean(pad_sequence[padding+n-rec_half_length:padding+n+rec_half_length+1])
            other_means = [np.mean(pad_sequence[padding+m-rec_half_length:padding+m+rec_half_length+1]) for m in range(n-rec_half_length, n+rec_half_length+1) if m != n]
            if all([n_mean >= o for o in other_means]) and n-n_last_peak > w:
                peaks.append(n)
                n_last_peak = n
    return peaks
            
def peakVisualization(ground_truth_peaks, predicted_sequence):
    print('eok')
    
    
def activationToEvents(activation_function, peak_thres, rec_half_length = 0, sr = 44100, frame_rate = 100):
    peaks = peakPickingRectangularByMean(activation_function, thres = peak_thres, rec_half_length = rec_half_length)
    peaks = np.array(peaks)
    events = peaks/frame_rate
    return events

def precisionRecallFmeasure(est_events, ref_events, est_pitches, ref_pitches, onset_tolerance = 0.05):
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
                break
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

def computeResults(dataset, y_hat_grouped, test_track_IDs, peak_thres, rec_half_length, FmWithBeats = False, soloDrum = None):
    # results dict initialization
    BD_results = {'precision': [], 'recall': [], 'fmeasure': []}
    SD_results = {'precision': [], 'recall': [], 'fmeasure': []}
    HH_results = {'precision': [], 'recall': [], 'fmeasure': []}
    beats_results = {'precision': [], 'recall': [], 'fmeasure': []}
    downbeats_results = {'precision': [], 'recall': [], 'fmeasure': []}
    global_results = {'precision': [], 'recall': [], 'fmeasure': []}
    
#    y_hat_grouped, test_track_IDs = groupePredictionSamplesByTrack(y_hat, test_IDs)
    
    for i, ID in enumerate(test_track_IDs):
#        print(i)
        
        # BD events
        if soloDrum == None or soloDrum == "BD":
            BD_est_activation = y_hat_grouped[i][:, 0]
            BD_est_events = activationToEvents(BD_est_activation, peak_thres, rec_half_length)
            BD_ref_events = np.array(dataset.data['BD_annotations'][ID])
            BD_est_pitches = np.ones(len(BD_est_events))
            BD_ref_pitches = np.ones(len(BD_ref_events))
            BD_precision, BD_recall, BD_fmeasure = f_measure(BD_est_events, BD_ref_events, BD_est_pitches, BD_ref_pitches)
            BD_results['precision'].append(BD_precision)
            BD_results['recall'].append(BD_recall)
            BD_results['fmeasure'].append(BD_fmeasure)
      
        # SD events
        if soloDrum == None or soloDrum == "SD":
            SD_est_activation = y_hat_grouped[i][:, 1]
            SD_est_events = activationToEvents(SD_est_activation, peak_thres, rec_half_length)
            SD_ref_events = np.array(dataset.data['SD_annotations'][ID])
            SD_est_pitches = np.ones(len(SD_est_events))
            SD_ref_pitches = np.ones(len(SD_ref_events))
            SD_precision, SD_recall, SD_fmeasure = f_measure(SD_est_events, SD_ref_events, SD_est_pitches, SD_ref_pitches)
            SD_results['precision'].append(SD_precision)
            SD_results['recall'].append(SD_recall)
            SD_results['fmeasure'].append(SD_fmeasure)
    
        # HH events
        if soloDrum == None or soloDrum == "HH":
            HH_est_activation = y_hat_grouped[i][:, 2]
            HH_est_events = activationToEvents(HH_est_activation, peak_thres, rec_half_length)
            HH_ref_events = np.array(dataset.data['HH_annotations'][ID])
            HH_est_pitches = np.ones(len(HH_est_events))
            HH_ref_pitches = np.ones(len(HH_ref_events))
            HH_precision, HH_recall, HH_fmeasure = f_measure(HH_est_events, HH_ref_events, HH_est_pitches, HH_ref_pitches)
            HH_results['precision'].append(HH_precision)
            HH_results['recall'].append(HH_recall)
            HH_results['fmeasure'].append(HH_fmeasure)
        
        if y_hat_grouped[i].shape[1] == 5:
            
            # beats events
            beats_est_activation = y_hat_grouped[i][:, 4]
            beats_est_events = activationToEvents(beats_est_activation, peak_thres, rec_half_length)
            beats_ref_events = np.array(dataset.data['beats_annotations'][ID])
            beats_est_pitches = np.ones(len(beats_est_events))
            beats_ref_pitches = np.ones(len(beats_ref_events))
            beats_precision, beats_recall, beats_fmeasure = f_measure(beats_est_events, beats_ref_events, beats_est_pitches, beats_ref_pitches)
            beats_results['precision'].append(beats_precision)
            beats_results['recall'].append(beats_recall)
            beats_results['fmeasure'].append(beats_fmeasure)
                        
            # downbeats events
            downbeats_est_activation = y_hat_grouped[i][:, 4]
            downbeats_est_events = activationToEvents(downbeats_est_activation, peak_thres, rec_half_length)
            downbeats_ref_events = np.array(dataset.data['downbeats_annotations'][ID])
            downbeats_est_pitches = np.ones(len(downbeats_est_events))
            downbeats_ref_pitches = np.ones(len(downbeats_ref_events))
            downbeats_precision, downbeats_recall, downbeats_fmeasure = f_measure(downbeats_est_events, downbeats_ref_events, downbeats_est_pitches, downbeats_ref_pitches)
            downbeats_results['precision'].append(downbeats_precision)
            downbeats_results['recall'].append(downbeats_recall)
            downbeats_results['fmeasure'].append(downbeats_fmeasure)            
            
    
        # all events
        if y_hat_grouped[0].shape[1] == 5 and FmWithBeats:
            all_est_events = np.concatenate((BD_est_events, SD_est_events, HH_est_events, beats_est_events, beats_est_events))
            all_est_pitches = np.concatenate((np.ones(len(BD_est_events)), np.ones(len(SD_est_events))*2, np.ones(len(HH_est_events))*3, np.ones(len(beats_est_events))*4, np.ones(len(downbeats_est_events))*5))
            all_ref_events = np.concatenate((BD_ref_events, SD_ref_events, HH_ref_events, beats_ref_events, downbeats_ref_events))
            all_ref_pitches = np.concatenate((np.ones(len(BD_ref_events)), np.ones(len(SD_ref_events))*2, np.ones(len(HH_ref_events))*3, np.ones(len(beats_ref_events))*4, np.ones(len(downbeats_ref_events))*5))
        else:
            all_est_events = np.concatenate((BD_est_events, SD_est_events, HH_est_events))
            all_est_pitches = np.concatenate((np.ones(len(BD_est_events)), np.ones(len(SD_est_events))*2, np.ones(len(HH_est_events))*3))
            all_ref_events = np.concatenate((BD_ref_events, SD_ref_events, HH_ref_events))
            all_ref_pitches = np.concatenate((np.ones(len(BD_ref_events)), np.ones(len(SD_ref_events))*2, np.ones(len(HH_ref_events))*3))

#        print(all_est_events.shape, all_est_pitches.shape, all_ref_events.shape, all_ref_pitches.shape)
        
        all_precision, all_recall, all_fmeasure = f_measure(all_est_events, all_ref_events, all_est_pitches, all_ref_pitches)
        global_results['precision'].append(all_precision)
        global_results['recall'].append(all_recall)
        global_results['fmeasure'].append(all_fmeasure)
        
    return BD_results, SD_results, HH_results, beats_results, downbeats_results, global_results
    

def visualizeModelPredictionPerTrack(test_track_ID, dataset, y_hat_grouped, test_track_IDs, BD_results, SD_results, HH_results, global_results, beats_results = None, downbeats_results = None):
    print(dataset.data['audio_name'][test_track_IDs[test_track_ID]])
    print("Bass drum: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(BD_results['precision'][test_track_ID], BD_results['recall'][test_track_ID], BD_results['fmeasure'][test_track_ID]))
    print("Snare drum: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(SD_results['precision'][test_track_ID], SD_results['recall'][test_track_ID], SD_results['fmeasure'][test_track_ID]))
    print("Hihat: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(HH_results['precision'][test_track_ID], HH_results['recall'][test_track_ID], HH_results['fmeasure'][test_track_ID]))
    if beats_results != None:
        print("Beats: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(beats_results['precision'][test_track_ID], beats_results['recall'][test_track_ID], beats_results['fmeasure'][test_track_ID]))
    if downbeats_results != None:
        print("Downbeats: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(downbeats_results['precision'][test_track_ID], downbeats_results['recall'][test_track_ID], downbeats_results['fmeasure'][test_track_ID]))

    print("Global: precision = {0:.3f} ; recall = {1:.3f} ; fmeasure = {2:.3f}".format(global_results['precision'][test_track_ID], global_results['recall'][test_track_ID], global_results['fmeasure'][test_track_ID]))
    
    if beats_results == None and downbeats_results == None:
        f, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    else:
        f, axes = plt.subplots(5, 1, sharex=True, sharey=True)

    # BD
    axes[0].plot(dataset.data['BD_target'][test_track_IDs[test_track_ID]])
    axes[0].plot(y_hat_grouped[test_track_ID][:, 0])
    axes[0].set_title('Activation function and ground-truth activation - Kick')
    
    # SD
    axes[1].plot(dataset.data['SD_target'][test_track_IDs[test_track_ID]])
    axes[1].plot(y_hat_grouped[test_track_ID][:, 1])
    axes[1].set_title('Activation function and ground-truth activation - Snare')
    
    # HH
    axes[2].plot(dataset.data['HH_target'][test_track_IDs[test_track_ID]])
    axes[2].plot(y_hat_grouped[test_track_ID][:, 2])
    axes[2].set_title('Activation function and ground-truth activation - Hihat')
    
    if beats_results != None and downbeats_results != None:
        # Beats
        axes[3].plot(dataset.data['beats_target'][test_track_IDs[test_track_ID]])
        axes[3].plot(y_hat_grouped[test_track_ID][:, 3])
        axes[3].set_title('Activation function and ground-truth activation - Beats')
        # Downbeats
        
        axes[4].plot(dataset.data['downbeats_target'][test_track_IDs[test_track_ID]])
        axes[4].plot(y_hat_grouped[test_track_ID][:, 4])
        axes[4].set_title('Activation function and ground-truth activation - Downbeats')