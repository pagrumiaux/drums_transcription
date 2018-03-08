#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:51:03 2018

@author: grumiaux
"""

import os
import numpy as np
from math import fmod, floor
import time
import xml.etree.ElementTree as et

folder_rbma = "/users/grumiaux/Documents/stage/rbma_13/"
folder_smt = "/users/grumiaux/Documents/stage/SMT_DRUMS/"

class Dataset:
    def __init__(self):
        self.data = {'audio_name': [], 'mel_spectrogram': [], 'BD_target': [], 'SD_target': [], 'HH_target': [], 'beats_target': [], 'downbeats_target': [], 'BD_annotations': [], 'SD_annotations': [], 'HH_annotations': [], 'beats_annotations': [], 'downbeats_annotations': []}
        
    def loadDataset(self):
        audio_names_rbma = self.extractAudioNamesRbma()
#        print(len(audio_names_rbma))
        for audio in audio_names_rbma:
#            print(audio)
            self.data['audio_name'].append(audio)
            mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsRbma(audio)
            self.data['mel_spectrogram'].append(mel_spectrogram)
            self.data['BD_annotations'].append(BD_annotations)
            self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
            self.data['SD_annotations'].append(SD_annotations)
            self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
            self.data['HH_annotations'].append(HH_annotations)
            self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
            self.data['beats_annotations'].append(beats_annotations)
            self.data['beats_target'].append(self.annotationsToTargetFunctions(beats_annotations, mel_spectrogram.shape[1]))
            self.data['downbeats_annotations'].append(downbeats_annotations)
            self.data['downbeats_target'].append(self.annotationsToTargetFunctions(downbeats_annotations, mel_spectrogram.shape[1]))
        print("Rbma dataset loaded.")
        
        audio_names_smt = self.extractAudioNamesSmt()
#        print(len(audio_names_smt))
        for audio in audio_names_smt:
            self.data['audio_name'].append(audio)
            mel_spectrogram, BD_annotations, SD_annotations, HH_annotations = self.extractMelSpectrogramAndAnnotationsSmt(audio)
            self.data['mel_spectrogram'].append(mel_spectrogram)
            self.data['BD_annotations'].append(BD_annotations)
            self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
            self.data['SD_annotations'].append(SD_annotations)
            self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
            self.data['HH_annotations'].append(HH_annotations)
            self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
        print('Smt dataset loaded.')
        
        self.audio_names = audio_names_rbma + audio_names_smt
        
    def extractAudioNamesRbma(self):
        folder_rbma_audio = folder_rbma + "annotations/beats"
        audio_names_rbma = [f[:-4] for f in os.listdir(folder_rbma_audio) if f.endswith('.txt')]
        return audio_names_rbma
    
    def extractAudioNamesSmt(self):
        folder_smt_audio = folder_smt + "data/mel"
        audio_names_smt_mix = [f[:-4] for f in os.listdir(folder_smt_audio) if f.endswith('MIX.npy')]
        audio_names_smt_other = [f[:-4] for f in os.listdir(folder_smt_audio) if f.endswith('.npy') and not f.endswith('MIX.npy')]
        audio_names_smt = audio_names_smt_mix + audio_names_smt_other
        return audio_names_smt
        
    def extractMelSpectrogramAndAnnotationsRbma(self, audio_name, sr = 44100):
        """ Compute the mel spectrogram of ONE track and extract the annotations
            input:
                name of the single audio track to extraction information without extension
            
        """
        # annotations variables initialization
        BD_annotations = []
        SD_annotations = []
        HH_annotations = []
        beats_annotations = []
        downbeats_annotations = []
        
        # mel spectrogram extraction
        mel_spectrogram = np.load(folder_rbma + "data/mel/" + audio_name + ".npy")
        
        # annotations extraction
        with open(folder_rbma + "annotations/drums/" + audio_name + ".txt", 'r') as f: # bass drum, snare drum and hihat annotations extraction
            lines = f.readlines()
            for line in lines:
                if int(line.split()[1]) == 0:
                    BD_annotations.append(max(float(line.split()[0]), 0))
                elif int(line.split()[1]) == 1:
                    SD_annotations.append(max(float(line.split()[0]), 0))
                elif int(line.split()[1]) == 2:
                    HH_annotations.append(max(float(line.split()[0]), 0))
        with open(folder_rbma + "annotations/beats/" + audio_name + ".txt", 'r') as f: # beats and downbeats annotations extraction
            lines = f.readlines()
            for line in lines:
                beats_annotations.append(float(line.split()[0]))
                if int(line.split()[1]) == 1:
                    downbeats_annotations.append(float(line.split()[0]))
        
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations
                         
    def extractMelSpectrogramAndAnnotationsSmt(self, audio_name, sr = 44100):
        # annotations variables initialization
        BD_annotations = []
        SD_annotations = []
        HH_annotations = []
        
        # mel spectrogram extraction
        mel_spectrogram = np.load(folder_smt + "data/mel/" + audio_name + ".npy")
        
        # annotations extraction
        if audio_name.endswith('KD') or audio_name.endswith('SD') or audio_name.endswith('HH'):
            xml_file = folder_smt + "annotation_xml/" + audio_name[:-2] + "MIX.xml"
        else:
            xml_file = folder_smt + "annotation_xml/" + audio_name + ".xml"
        tree = et.parse(xml_file)
        for event in tree.iter('event'):
            instrument = event[3]
            if instrument.text == 'KD':
                BD_annotations.append(max(float(event[1].text), 0))
            elif instrument.text == 'SD':
                SD_annotations.append(max(float(event[1].text), 0))
            elif instrument.text == 'HH':
                HH_annotations.append(max(float(event[1].text), 0))
        
        if audio_name.endswith("KD"):
            SD_annotations = []
            HH_annotations = []
        elif audio_name.endswith("SD"):
            BD_annotations = []
            HH_annotations = []
        elif audio_name.endswith("HH"):
            BD_annotations = []
            SD_annotations = []
                
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations
    
    def annotationsToTargetFunctions(self, annotations, n_frames, sr = 44100, frame_rate = 100):
        """ Transform annotations list into target functions
            input:
                annotations : list of float that represents the positions in seconds
                signal_length : length of the signal for the target function creation
                frame_rate : number of frame per second for the target function
        """
    #    target_function = np.zeros(floor(signal_length/(sr/frame_rate)+1))
        target_function = np.zeros(n_frames)
        for item in annotations:
            frame_number = int(frame_rate*item)
            if frame_number < len(target_function):
                target_function[frame_number] = 1        
        return target_function
    
    def generate_IDs(self, task, context_frames = 25, sequential_frames = 100, withBeatsAnnotations = False, dataFilter = None):
        list_IDs = []
        n_audio = len(self.data['mel_spectrogram'])
        for i in range(n_audio):
            n_frames = self.data['mel_spectrogram'][i].shape[1]
            if task == 'CNN':
                for j in range(n_frames):
                    list_IDs.append((i, j))
            elif task == 'RNN':
                for j in range(n_frames):
                    if fmod(j, 100) == 0:
                        list_IDs.append((i, j))
                    
        if withBeatsAnnotations == True:
            list_IDs = [ID for ID in list_IDs if ID[0] < 27]
            
        if dataFilter != None:
            if dataFilter == 'rbma':
                list_IDs = [ID for ID in list_IDs if ID[0] < 27]
            elif dataFilter == 'smt':
                list_IDs = [ID for ID in list_IDs if ID[0] >= 27]
        return list_IDs

