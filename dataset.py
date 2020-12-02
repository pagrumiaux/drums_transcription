#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:51:03 2018

@author: grumiaux
"""

import os
import numpy as np
from math import fmod, floor
import xml.etree.ElementTree as et
from utilities import spreadTargetFunctions
import pickle


class Dataset:
    def __init__(self, folder_rbma, folder_smt, folder_enst):
        self.data = {'audio_name': [], 'mel_spectrogram': [], 'dataset': [],
                     'BD_target': [], 'SD_target': [], 'HH_target': [],
                     'beats_target': [], 'downbeats_target': [],
                     'BD_annotations': [], 'SD_annotations': [],
                     'HH_annotations': [], 'beats_annotations': [],
                     'downbeats_annotations': []
                    }
        self.split = {'rbma_train_IDs': [], 'rbma_test_IDs': [],
                      'smt_train_IDs': [], 'smt_test_IDs':[]
                     }
        self.standardization = {}
        
        self.folder_rbma = folder_rbma
        self.folder_smt = folder_smt
        self.folder_enst = folder_enst
    
    def load_dataset(self, enst_solo, bb_annotations_folder = None, spread_length = None):    
        
        # RBMA load
        self.load_data('rbma')
        print("RBMA dataset loaded.")
        
        # SMT load
        self.load_data('smt')
        print("SMT dataset loaded.")
        
        # ENST load
        self.load_data('enst', enst_solo)
        print("ENST dataset loaded.")
        
        
        # # RBMA load
        # audio_names_rbma = self.extractAudioNamesRbma()
        # for audio in audio_names_rbma:
        #     self.data['audio_name'].append(audio)
        #     mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsRbma(audio)
        #     self.data['mel_spectrogram'].append(mel_spectrogram)
        #     self.data['dataset'].append('rbma')
        #     self.data['BD_annotations'].append(BD_annotations)
        #     self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
        #     self.data['SD_annotations'].append(SD_annotations)
        #     self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
        #     self.data['HH_annotations'].append(HH_annotations)
        #     self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
        #     self.data['beats_annotations'].append(beats_annotations)
        #     self.data['beats_target'].append(self.annotationsToTargetFunctions(beats_annotations, mel_spectrogram.shape[1]))
        #     self.data['downbeats_annotations'].append(downbeats_annotations)
        #     self.data['downbeats_target'].append(self.annotationsToTargetFunctions(downbeats_annotations, mel_spectrogram.shape[1]))
            
        # with open(self.folder_rbma + "data/other/train_IDs", "rb") as f:
        #     self.split['rbma_train_IDs'] = pickle.load(f)
        # with open(self.folder_rbma + "data/other/test_IDs", "rb") as f:
        #     self.split['rbma_test_IDs'] = pickle.load(f)
        # with open(self.folder_rbma + "data/other/train_mel_mean", "rb") as f:
        #     self.standardization['rbma_mel_mean'] = pickle.load(f)   
        # with open(self.folder_rbma + "data/other/train_mel_var", "rb") as f:
        #     self.standardization['rbma_mel_var'] = pickle.load(f)
        
        # print("Rbma dataset loaded.")
        
        # # SMT load
        # audio_names_smt = self.extractAudioNamesSmt()
        # for audio in audio_names_smt:
        #     self.data['audio_name'].append(audio)
        #     mel_spectrogram, BD_annotations, SD_annotations, HH_annotations = self.extractMelSpectrogramAndAnnotationsSmt(audio)
        #     self.data['mel_spectrogram'].append(mel_spectrogram)
        #     self.data['dataset'].append('smt')
        #     self.data['BD_annotations'].append(BD_annotations)
        #     self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
        #     self.data['SD_annotations'].append(SD_annotations)
        #     self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
        #     self.data['HH_annotations'].append(HH_annotations)
        #     self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
        #     self.data['beats_annotations'].append(None)
        #     self.data['beats_target'].append(None)
        #     self.data['downbeats_annotations'].append(None)
        #     self.data['downbeats_target'].append(None)

        
        # with open(folder_smt + "data/other/train_IDs", "rb") as f:
        #     self.split['smt_train_IDs'] = pickle.load(f)
        # with open(folder_smt + "data/other/test_IDs", "rb") as f:
        #     self.split['smt_test_IDs'] = pickle.load(f)
        # with open(folder_smt + "data/other/train_mel_mean", "rb") as f:
        #     self.standardization['smt_mel_mean'] = pickle.load(f)   
        # with open(folder_smt + "data/other/train_mel_var", "rb") as f:
        #     self.standardization['smt_mel_var'] = pickle.load(f)
        
        # print('Smt dataset loaded.')
        
        # # ENST load
        # audio_names_enst = self.extractAudioNamesEnst()
        # for audio in audio_names_enst:
        #     self.data['audio_name'].append(audio)
        #     mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsEnst(audio, enst_solo)
        #     self.data['mel_spectrogram'].append(mel_spectrogram)
        #     self.data['dataset'].append('enst')
        #     self.data['BD_annotations'].append(BD_annotations)
        #     self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
        #     self.data['SD_annotations'].append(SD_annotations)
        #     self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
        #     self.data['HH_annotations'].append(HH_annotations)
        #     self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
        #     self.data['beats_annotations'].append(beats_annotations)
        #     self.data['beats_target'].append(self.annotationsToTargetFunctions(beats_annotations, mel_spectrogram.shape[1]))
        #     self.data['downbeats_annotations'].append(downbeats_annotations)
        #     self.data['downbeats_target'].append(self.annotationsToTargetFunctions(downbeats_annotations, mel_spectrogram.shape[1]))
            
        # with open(folder_enst + "data/other/train_IDs", "rb") as f:
        #     self.split['enst_train_IDs'] = pickle.load(f)
        # with open(folder_enst + "data/other/test_IDs", "rb") as f:
        #     self.split['enst_test_IDs'] = pickle.load(f)
        # if not enst_solo:
        #     with open(folder_enst + "data/other/train_mel_mean", "rb") as f:
        #         self.standardization['enst_mel_mean'] = pickle.load(f)   
        #     with open(folder_enst + "data/other/train_mel_var", "rb") as f:
        #         self.standardization['enst_mel_var'] = pickle.load(f)
        # else:
        #     with open(folder_enst + "data/other/train_mel_mean_solo", "rb") as f:
        #         self.standardization['enst_mel_mean'] = pickle.load(f)   
        #     with open(folder_enst + "data/other/train_mel_var_solo", "rb") as f:
        #         self.standardization['enst_mel_var'] = pickle.load(f)
            
        # print("Enst dataset loaded.")


        # # Billboard load
        # audio_names_bb = self.extractAudioNamesBillboard()
        # for audio in audio_names_bb:
        #     self.data['audio_name'].append(audio)
        #     mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, donwbeats_annotations = self.extractMelSpectrogramAndAnnotationsBillboard(audio, bb_annotations_folder)
        #     self.data['mel_spectrogram'].append(mel_spectrogram)
        #     self.data['dataset'].append('billboard')
        #     self.data['BD_annotations'].append(BD_annotations)
        #     self.data['BD_target'].append(self.annotationsToTargetFunctions(BD_annotations, mel_spectrogram.shape[1]))
        #     self.data['SD_annotations'].append(SD_annotations)
        #     self.data['SD_target'].append(self.annotationsToTargetFunctions(SD_annotations, mel_spectrogram.shape[1]))
        #     self.data['HH_annotations'].append(HH_annotations)
        #     self.data['HH_target'].append(self.annotationsToTargetFunctions(HH_annotations, mel_spectrogram.shape[1]))
        #     self.data['beats_annotations'].append(beats_annotations)
        #     self.data['beats_target'].append(self.annotationsToTargetFunctions(beats_annotations, mel_spectrogram.shape[1]))
        #     self.data['downbeats_annotations'].append(downbeats_annotations)
        #     self.data['downbeats_target'].append(self.annotationsToTargetFunctions(downbeats_annotations, mel_spectrogram.shape[1]))
            
        # self.split['bb_train_IDs'] = list(range(800))
        # self.split['bb_test_IDs'] = []
        
        # print("Billboard dataset loaded.")
        
        # we spread the annotation 1.0
        if spread_length != None:
            nb_target_functions = len(self.data['BD_target'])
            for i in range(nb_target_functions):
                if self.data['dataset'][i] != 'billboard':
                    self.data['BD_target'][i] = spreadTargetFunctions(self.data['BD_target'][i], spread_length)
                    self.data['SD_target'][i] = spreadTargetFunctions(self.data['SD_target'][i], spread_length)
                    self.data['HH_target'][i] = spreadTargetFunctions(self.data['HH_target'][i], spread_length)
                    if i <= 26 or i >= 311:
                        self.data['beats_target'][i] = spreadTargetFunctions(self.data['beats_target'][i], spread_length)
                        self.data['downbeats_target'][i] = spreadTargetFunctions(self.data['downbeats_target'][i], spread_length)
            print('Spreading done over all samples')
        
        # self.audio_names = audio_names_rbma + audio_names_smt
    
    def load_data(self, dataset_name, enst_solo=False):
        
        # Verify if dataset_name is correct
        if dataset_name not in ['rbma', 'enst', 'smt']:
            raise ValueError("Unknown dataset name. It should be one of the \
                             following names: 'rbma', 'smt', 'enst'.")
        
        
        # Extract audio file names within the dataset folder
        audio_names = self.extract_audio_names(dataset_name)
        
        # Extract annotation for all audio files
        for audio in audio_names:
            self.data['audio_name'].append(audio)
            
            # Extract spectrograms and annotations within the dataset folder
            if dataset_name == 'rbma':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsRbma(audio)
            elif dataset_name == 'smt':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsSmt(audio)
            elif dataset_name == 'enst':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extractMelSpectrogramAndAnnotationsEnst(audio)
            
            # Load data
            self.data['mel_spectrogram'].append(mel_spectrogram)
            self.data['dataset'].append('rbma')
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
            
        # Load training/validation/test splits and standardization mean and var values in the dataset folder
        if dataset_name == 'rbma':
            # RBMA dataset
            with open(self.folder_rbma + "data/other/train_IDs", "rb") as f:
                self.split['rbma_train_IDs'] = pickle.load(f)
            with open(self.folder_rbma + "data/other/test_IDs", "rb") as f:
                self.split['rbma_test_IDs'] = pickle.load(f)
            with open(self.folder_rbma + "data/other/train_mel_mean", "rb") as f:
                self.standardization['rbma_mel_mean'] = pickle.load(f)   
            with open(self.folder_rbma + "data/other/train_mel_var", "rb") as f:
                self.standardization['rbma_mel_var'] = pickle.load(f)
                
        elif dataset_name == 'smt':
            # SMT dataset
            with open(self.folder_smt + "data/other/train_IDs", "rb") as f:
                self.split['smt_train_IDs'] = pickle.load(f)
            with open(self.folder_smt + "data/other/test_IDs", "rb") as f:
                self.split['smt_test_IDs'] = pickle.load(f)
            with open(self.folder_smt + "data/other/train_mel_mean", "rb") as f:
                self.standardization['smt_mel_mean'] = pickle.load(f)   
            with open(self.folder_smt + "data/other/train_mel_var", "rb") as f:
                self.standardization['smt_mel_var'] = pickle.load(f)
                
        elif dataset_name == 'enst':
            # ENST dataset
            with open(self.folder_enst + "data/other/train_IDs", "rb") as f:
                self.split['enst_train_IDs'] = pickle.load(f)
            with open(self.folder_enst + "data/other/test_IDs", "rb") as f:
                self.split['enst_test_IDs'] = pickle.load(f)
            if enst_solo:
                with open(self.folder_enst + "data/other/train_mel_mean_solo", "rb") as f:
                    self.standardization['enst_mel_mean'] = pickle.load(f)   
                with open(self.folder_enst + "data/other/train_mel_var_solo", "rb") as f:
                    self.standardization['enst_mel_var'] = pickle.load(f)
            else:
                with open(self.folder_enst + "data/other/train_mel_mean", "rb") as f:
                    self.standardization['enst_mel_mean'] = pickle.load(f)   
                with open(self.folder_enst + "data/other/train_mel_var", "rb") as f:
                    self.standardization['enst_mel_var'] = pickle.load(f)

    def extract_audio_names(self, dataset_name):     
        if dataset_name == 'rbma':
            if not os.path.isdir(self.folder_rbma):
                raise ValueError(f'RBMA folder not found : {self.folder_rbma}')
            folder_rbma_audio = self.folder_rbma + "annotations/beats"
            audio_names = [f[:-4] for f in os.listdir(folder_rbma_audio) 
                           if f.endswith('.txt')]
            audio_names = sorted(audio_names)
            
        elif dataset_name == 'smt':
            if not os.path.isdir(self.folder_smt):
                raise ValueError(f'SMT folder not found : {self.folder_smt}')
            folder_smt_audio = self.folder_smt + "data/log_mel"
            audio_names_smt_mix = [f[:-4] for f in os.listdir(folder_smt_audio) 
                                   if f.endswith('MIX.npy')]
            audio_names_smt_mix = sorted(audio_names_smt_mix)
            audio_names_smt_other = [f[:-4] for f in os.listdir(folder_smt_audio) 
                                     if f.endswith('.npy') 
                                     and not f.endswith('MIX.npy')]
            audio_names_smt_other = sorted(audio_names_smt_other)
            audio_names = audio_names_smt_mix + audio_names_smt_other
            
        elif dataset_name == 'enst':
            if not os.path.isdir(self.folder_enst):
                raise ValueError(f'ENST folder not found : {self.folder_enst}')
            folder_enst_audio = self.folder_enst + "annotations/drums/"
            audio_names = [f[:-4] for f in os.listdir(folder_enst_audio) 
                           if f.endswith('.txt')]
            audio_names = sorted(audio_names)
        
        return audio_names
    
    # def extractAudioNamesBillboard(self):
    #     folder_bb_data = folder_bb + "data/log_mel/"
    #     audio_names_bb = [f[:-4] for f in os.listdir(folder_bb_data) 
    #                       if f.endswith('.npy')]
    #     audio_names_bb = sorted(audio_names_bb)
    #     return audio_names_bb
        
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
        mel_spectrogram = np.load(self.folder_rbma + "data/log_mel/" + audio_name + ".npy")
        
        # annotations extraction
        with open(self.folder_rbma + "annotations/drums/" + audio_name + ".txt", 'r') as f: # bass drum, snare drum and hihat annotations extraction
            lines = f.readlines()
            for line in lines:
                if int(line.split()[1]) == 0:
                    BD_annotations.append(max(float(line.split()[0]), 0))
                elif int(line.split()[1]) == 1:
                    SD_annotations.append(max(float(line.split()[0]), 0))
                elif int(line.split()[1]) == 2:
                    HH_annotations.append(max(float(line.split()[0]), 0))
        with open(self.folder_rbma + "annotations/beats_madmom/" + audio_name + ".txt", 'r') as f: # beats and downbeats annotations extraction
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
        mel_spectrogram = np.load(self.folder_smt + "data/log_mel/" + audio_name + ".npy")
        
        # annotations extraction
        if audio_name.endswith('KD') or audio_name.endswith('SD') or audio_name.endswith('HH'):
            xml_file = self.folder_smt + "annotation_xml/" + audio_name[:-2] + "MIX.xml"
        else:
            xml_file = self.folder_smt + "annotation_xml/" + audio_name + ".xml"
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
    
    def extractMelSpectrogramAndAnnotationsEnst(self, audio_name, enst_solo, sr = 44100):
        # annotations variables initialization
        BD_annotations = []
        SD_annotations = []
        HH_annotations = []
        beats_annotations = []
        downbeats_annotations = []
        
        # mel spectrogram extraction
        if enst_solo:
            mel_spectrogram = np.load(self.folder_enst + "data/log_mel/drums/" + audio_name + ".npy")
        else:
            mel_spectrogram = np.load(self.folder_enst + "data/log_mel/mix66/" + audio_name + ".npy")
        
                # annotations extraction
        with open(self.folder_enst + "annotations/drums/" + audio_name + ".txt", 'r') as f: # bass drum, snare drum and hihat annotations extraction
            lines = f.readlines()
            for line in lines:
                if line.split()[1] == 'bd':
                    BD_annotations.append(max(float(line.split()[0]), 0))
                elif line.split()[1] == 'sd' or line.split()[1] == 'sd-':
                    SD_annotations.append(max(float(line.split()[0]), 0))
                elif line.split()[1] == 'ooh' or line.split()[1] == 'chh':
                    HH_annotations.append(max(float(line.split()[0]), 0))   
        
        xml_file = self.folder_enst + "annotations/beats/" + audio_name + ".xml"
        tree = et.parse(xml_file)
        root = tree.getroot()
        for seg in root.iter('{http://www.ircam.fr/musicdescription/1.1}segment'):
            t = seg.get('time')
#            print(type(t))
            beattype = seg.find('{http://www.ircam.fr/musicdescription/1.1}beattype')
            db = beattype.get('measure')
            beats_annotations.append(max(float(t), 0))
            if db == '1':
                downbeats_annotations.append(max(float(t), 0))     
            
        
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations
    
    def extractMelSpectrogramAndAnnotationsBillboard(self, audio_name, bb_annotations_folder, sr = 44100):
        # annotations variables initialization
        mel_spectrogram = np.load(folder_bb + "data/log_mel/" + audio_name + ".npy")
        if bb_annotations_folder is not None:
            with open('./billboard/target/' + bb_annotations_folder + '/hard/' + audio_name + '_BD', 'rb') as f:
                BD_annotations = pickle.load(f)
            with open('./billboard/target/' + bb_annotations_folder + '/hard/' + audio_name + '_SD', 'rb') as f:
                SD_annotations = pickle.load(f)
            with open('./billboard/target/' + bb_annotations_folder + '/hard/' + audio_name + '_HH', 'rb') as f:
                HH_annotations = pickle.load(f)
            with open('./billboard/annotations/beats/' + audio_name + '.txt', 'r') as f:
                lines = f.readlines()
                beats_annotations = []
                downbeats_annotations = []
                for line in lines:
                    (t, b) = line.split()
                    beats_annotations.append(float(t)-30.0)
                    if b == '1':
                        beats_annotations.append(float(t)-30.0)                        
        else:
            BD_annotations = None
            SD_annotations = None
            HH_annotations = None    
            beats_annotations = None
            downbeats_annotations = None
        
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations

    
    def annotationsToTargetFunctions(self, annotations, n_frames, sr = 44100, frame_rate = 100):
        """ Transform annotations list into target functions
            input:
                annotations : list of float that represents the positions in seconds
                signal_length : length of the signal for the target function creation
                frame_rate : number of frame per second for the target function
        """
        if annotations is not None:
            target_function = np.zeros(n_frames)
            for item in annotations:
                frame_number = int(frame_rate*item)
                if frame_number < len(target_function):
                    target_function[frame_number] = 1        
            return target_function
        else:
            return None
    
    def generate_IDs(self, task, stride = 0, context_frames = 25, sequential_frames = 100, dataFilter = None):
        list_IDs = []
        n_audio = len(self.data['mel_spectrogram'])
        for i in range(n_audio):
            n_frames = self.data['mel_spectrogram'][i].shape[1]
            if task == 'CNN':
                for j in range(n_frames):
                    if stride > 0:
                        if fmod(j, stride) == 0:
                            list_IDs.append((i, j))
                    else:
                        list_IDs.append((i, j))
            elif task == 'RNN' or task == 'CBRNN':
                for j in range(n_frames):
                    if fmod(j, sequential_frames) == 0:
                        list_IDs.append((i, j))
#                    list_IDs.append((i, j))
            elif task == 'DNN':
                for j in range(n_frames):
                    list_IDs.append((i, j))
            
        if dataFilter != None:
            if dataFilter == 'rbma':
                list_IDs = [ID for ID in list_IDs if ID[0] <= 26]
            elif dataFilter == 'smt':
                list_IDs = [ID for ID in list_IDs if ID[0] >= 27 and ID[0] <= 310]
            elif dataFilter == 'enst':
                list_IDs = [ID for ID in list_IDs if ID[0] >= 311 and ID[0] <= 373]
            elif dataFilter == 'bb':
                list_IDs = [ID for ID in list_IDs if ID[0] >= 374]
        return list_IDs

