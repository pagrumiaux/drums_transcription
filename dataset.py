# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:51:03 2018

@author: grumiaux
"""

import os
import numpy as np
from math import fmod, floor
import xml.etree.ElementTree as et
from utils.utilities import spreadTargetFunctions
import pickle

class Dataset:
    """A class representing a dataset. It loads the data without processing
    and is useful for visualizing the data, the target, the annotations, etc.
    It also splits the data into training and test sets.

    Attributes
    ----------
    data : dict
        Dictionnary containing all the data, stored in lists.
        Corresponding fields for the same data are at the same 
        position in each field list.
    split : dict
        Dictionnary containing the IDs split into train and test sets,
        and separated according to the initial datasets.
    standardization : dict
        Dictionnary containing the mean and standard deviation arrays
        for input standardization
    folder_rbma : str
        Path to the folder containing RBMA 13 dataset
    folder_smt : str
        Path to the folder containing SMT-Drums dataset
    folder_enst : str
        Path to the folder containing ENST-Drums dataset
    enst_solo : bool
        UsIf True, we use only the drums part of the tracks
        in the ENST dataset, otherwise we use the mixed tracks.

    Methods
    -------
    load_dataset(bb_annotations_folder = None, spread_length = None)
        Load all the data from the specified folders and store it into
        self.data dictionnary. If a folder path attributes is not specified,
        data is not loaded.
    load_data(self, dataset_name, enst_solo=False)
        Load the filenames, spectrograms, annotations, splits and standardization arrays, from a specific dataset folder. 
    extract_audio_names(dataset_name)
        Extract the audio filenames for a specific dataset folder.
    extract_data_rbma(audio_name)
        Extract the spectrograms and annotations for RBMA 13 dataset.
    extract_data_smt(audio_name)
        Extract the spectrograms and annotations for SMT-Drums dataset.
    extract_data_enst(audio_name)
        Extract the spectrograms and annotations for ENST-Drums dataset.
    extract_data_bb(audio_name, bb_annotations_folder)
        Extract the spectrograms and annotations for Billboard 800 dataset.
    annotation_to_target(annotations, n_frames, frame_rate = 100)
        Transform the extracted annotations into target for the training.
    generate_IDs(task, stride = 0, context_frames = 25, sequential_frames = 100)
        Generate the training and testing examples IDs.
    """
    def __init__(self, folder_rbma=None, folder_smt=None, folder_enst=None, enst_solo=False):
        self.data = {'audio_name': [], 'mel_spectrogram': [], 'dataset': [],
                     'BD_target': [], 'SD_target': [], 'HH_target': [],
                     'beats_target': [], 'downbeats_target': [],
                     'BD_annotations': [], 'SD_annotations': [],
                     'HH_annotations': [], 'beats_annotations': [],
                     'downbeats_annotations': []
                    }
        self.split = {'rbma_train_files': [], 'rbma_test_files': [],
                      'smt_train_files': [], 'smt_test_files': [],
                      'enst_train_files': [], 'enst_test_files': []
                     }
        self.standardization = {}
        
        self.folder_rbma = folder_rbma
        self.folder_smt = folder_smt
        self.folder_enst = folder_enst
        
        self.enst_solo = enst_solo
        
        self.load_dataset()
        
    def __len__(self):
        return len(self.data['audio_name'])
    
    def load_dataset(self, bb_annotations_folder = None, spread_length = None):    
        """Load the data from the dataset folder (folder_rbma, folder_smt
        and folder_enst) and put them into self.data dictionnary.

        Parameters
        ----------
        bb_annotations_folder : [type], optional TODO
            [description], by default None
        spread_length : [type], optional
            [description], by default None
        """
        # RBMA load
        if self.folder_rbma is not None:
            self.load_data('rbma')
            print("RBMA dataset loaded.")
        
        # SMT load
        if self.folder_smt is not None:
            self.load_data('smt')
            print("SMT dataset loaded.")
        
        # ENST load
        if self.folder_enst is not None:
            self.load_data('enst', self.enst_solo)
            print("ENST dataset loaded.")


        # # Billboard load
        # audio_names_bb = self.extractAudioNamesBillboard()
        # for audio in audio_names_bb:
        #     self.data['audio_name'].append(audio)
        #     mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, donwbeats_annotations = self.extract_data_bb(audio, bb_annotations_folder)
        #     self.data['mel_spectrogram'].append(mel_spectrogram)
        #     self.data['dataset'].append('billboard')
        #     self.data['BD_annotations'].append(BD_annotations)
        #     self.data['BD_target'].append(self.annotation_to_target(BD_annotations, mel_spectrogram.shape[1]))
        #     self.data['SD_annotations'].append(SD_annotations)
        #     self.data['SD_target'].append(self.annotation_to_target(SD_annotations, mel_spectrogram.shape[1]))
        #     self.data['HH_annotations'].append(HH_annotations)
        #     self.data['HH_target'].append(self.annotation_to_target(HH_annotations, mel_spectrogram.shape[1]))
        #     self.data['beats_annotations'].append(beats_annotations)
        #     self.data['beats_target'].append(self.annotation_to_target(beats_annotations, mel_spectrogram.shape[1]))
        #     self.data['downbeats_annotations'].append(downbeats_annotations)
        #     self.data['downbeats_target'].append(self.annotation_to_target(downbeats_annotations, mel_spectrogram.shape[1]))
            
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
        """Intermediary function to load data specifically for one dataset.

        Parameters
        ----------
        dataset_name : str
            dataset name : 'rbma', 'smt' or 'enst'
        enst_solo : bool, optional
            Use solo part of the ENST dataset or not, by default False
        """        
        # Verify if dataset_name is correct
        if dataset_name not in ['rbma', 'enst', 'smt']:
            raise ValueError("Unknown dataset name. It should be one of the \
                             following names: 'rbma', 'smt', 'enst'.")
        
        
        # Extract audio file names within the dataset folder
        audio_names = self.extract_audio_names(dataset_name)
        
        # Extract data and annotation for all audio files
        for audio in audio_names:
            self.data['audio_name'].append(audio)
            
            # Extract spectrograms and annotations within the dataset folder
            if dataset_name == 'rbma':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extract_data_rbma(audio)
            elif dataset_name == 'smt':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extract_data_smt(audio)
            elif dataset_name == 'enst':
                mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations = self.extract_data_enst(audio)
            
            # Load data
            self.data['mel_spectrogram'].append(mel_spectrogram)
            self.data['dataset'].append('rbma')
            self.data['BD_annotations'].append(BD_annotations)
            self.data['BD_target'].append(self.annotation_to_target(BD_annotations, mel_spectrogram.shape[1]))
            self.data['SD_annotations'].append(SD_annotations)
            self.data['SD_target'].append(self.annotation_to_target(SD_annotations, mel_spectrogram.shape[1]))
            self.data['HH_annotations'].append(HH_annotations)
            self.data['HH_target'].append(self.annotation_to_target(HH_annotations, mel_spectrogram.shape[1]))
            self.data['beats_annotations'].append(beats_annotations)
            self.data['beats_target'].append(self.annotation_to_target(beats_annotations, mel_spectrogram.shape[1]))
            self.data['downbeats_annotations'].append(downbeats_annotations)
            self.data['downbeats_target'].append(self.annotation_to_target(downbeats_annotations, mel_spectrogram.shape[1]))
            
        # Load training/validation/test splits and standardization mean and var values in the dataset folder
        if dataset_name == 'rbma':
            # RBMA dataset
            with open(self.folder_rbma + "split/train_files", "rb") as f:
                self.split['rbma_train_files'] = pickle.load(f)
            with open(self.folder_rbma + "split/test_files", "rb") as f:
                self.split['rbma_test_files'] = pickle.load(f)
            with open(self.folder_rbma + "data/other/train_mel_mean", "rb") as f:
                self.standardization['rbma_mel_mean'] = pickle.load(f)   
            with open(self.folder_rbma + "data/other/train_mel_var", "rb") as f:
                self.standardization['rbma_mel_var'] = pickle.load(f)
                
        elif dataset_name == 'smt':
            # SMT dataset
            with open(self.folder_smt + "split/train_files", "rb") as f:
                self.split['smt_train_files'] = pickle.load(f)
            with open(self.folder_smt + "split/test_files", "rb") as f:
                self.split['smt_test_files'] = pickle.load(f)
            with open(self.folder_smt + "data/other/train_mel_mean", "rb") as f:
                self.standardization['smt_mel_mean'] = pickle.load(f)   
            with open(self.folder_smt + "data/other/train_mel_var", "rb") as f:
                self.standardization['smt_mel_var'] = pickle.load(f)
                
        elif dataset_name == 'enst':
            # ENST dataset
            with open(self.folder_enst + "split/train_files", "rb") as f:
                self.split['enst_train_files'] = pickle.load(f)
            with open(self.folder_enst + "split/test_files", "rb") as f:
                self.split['enst_test_files'] = pickle.load(f)
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
        """Extract the filename of the tracks in the specified dataset folder.

        Parameters
        ----------
        dataset_name : str
            Dataset name, must be 'rbma', 'smt' or 'enst'.

        Returns
        -------
        list of str
            List of the filenames without extension.
        """             
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
        
    def extract_data_rbma(self, audio_name, sr = 44100):
        """Compute the mel spectrogram of ONE track and extract the annotations
        for RBMA 13 dataset

        Parameters
        ----------
        audio_name : str
            Filename to extract data
        sr : int, optional
            sampling rate, by default 44100

        Returns
        -------
        mel_spectrogram : np.array
            Numpy array representing the mel spectrogram
        BD_annotations : list of float
            Contains the timestamps of all bass drum hits in the track
        SD_annotations : list of float
            Contains the timestamps of all snare drum hits in the track
        HH_annotations : list of float
            Contains the timestamps of all hi-hat hits in the track
        beats_annotations : list of float
            Contains the timestamps of all the beats in the track
        downbeats_annotations : list of float
            Contains the timestamps of all the downbeats in the track
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
                         
    def extract_data_smt(self, audio_name):
        """Compute the mel spectrogram of ONE track and extract the annotations
        for SMT-Drums dataset

        Parameters
        ----------
        audio_name : str
            Filename to extract data

        Returns
        -------
        mel_spectrogram : np.array
            Numpy array representing the mel spectrogram
        BD_annotations : list of float
            Contains the timestamps of all bass drum hits in the track
        SD_annotations : list of float
            Contains the timestamps of all snare drum hits in the track
        HH_annotations : list of float
            Contains the timestamps of all hi-hat hits in the track
        beats_annotations : list of float
            Contains the timestamps of all the beats in the track
        downbeats_annotations : list of float
            Contains the timestamps of all the downbeats in the track
        """   

        # annotations variables initialization
        BD_annotations = []
        SD_annotations = []
        HH_annotations = []
        beats_annotations = []
        downbeats_annotations = []
        
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
                
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations
    
    def extract_data_enst(self, audio_name):
        """Compute the mel spectrogram of ONE track and extract the annotations
        for ENST-Drums dataset

        Parameters
        ----------
        audio_name : str
            Filename to extract data

        Returns
        -------
        mel_spectrogram : np.array
            Numpy array representing the mel spectrogram
        BD_annotations : list of float
            Contains the timestamps of all bass drum hits in the track
        SD_annotations : list of float
            Contains the timestamps of all snare drum hits in the track
        HH_annotations : list of float
            Contains the timestamps of all hi-hat hits in the track
        beats_annotations : list of float
            Contains the timestamps of all the beats in the track
        downbeats_annotations : list of float
            Contains the timestamps of all the downbeats in the track
        """

        # annotations variables initialization
        BD_annotations = []
        SD_annotations = []
        HH_annotations = []
        beats_annotations = []
        downbeats_annotations = []
        
        # mel spectrogram extraction
        if self.enst_solo:
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

            beattype = seg.find('{http://www.ircam.fr/musicdescription/1.1}beattype')
            db = beattype.get('measure')
            beats_annotations.append(max(float(t), 0))
            if db == '1':
                downbeats_annotations.append(max(float(t), 0))     
            
        
        return mel_spectrogram, BD_annotations, SD_annotations, HH_annotations, beats_annotations, downbeats_annotations
    
    def extract_data_bb(self, audio_name, bb_annotations_folder):
        # annotations variables initialization
        mel_spectrogram = np.load(self.folder_bb + "data/log_mel/" + audio_name + ".npy")
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

    
    def annotation_to_target(self, annotations, n_frames, frame_rate = 100):
        """Transform annotation lists into target format for the neural 
        networks

        Parameters
        ----------
        annotations : list of float
            Annotation list containing all the timestamps
        n_frames : int
            total number of annotation frames
        frame_rate : int, optional
            Frame per second, by default 100

        Returns
        -------
        [type]
            [description]
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
    
    def generate_IDs(self, task, stride = 0, context_frames = 25, sequential_frames = 100):
        """ Generate the IDs depending on several training parameters.

        Parameters
        ----------
        task : str
            type of neural network we use in the training : 'cnn', 'rnn' or 'crnn'
        stride : int, optional
            frame hop size, by default 0
        context_frames : int, optional
            Number of context frames for 'cnn' and 'crnn' tasks, by default 25
        sequential_frames : int, optional
            Number of frames in a sequence for 'rnn' and 'crnn', by default 100

        Returns
        -------
        list of int
            list of IDs for training
        """    
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
            elif task == 'RNN' or task == 'crnn':
                for j in range(n_frames):
                    if fmod(j, sequential_frames) == 0:
                        list_IDs.append((i, j))
            elif task == 'DNN':
                for j in range(n_frames):
                    list_IDs.append((i, j))
            
        # if dataFilter != None:
        #     if dataFilter == 'rbma':
        #         list_IDs = [ID for ID in list_IDs if ID[0] <= 26]
        #     elif dataFilter == 'smt':
        #         list_IDs = [ID for ID in list_IDs if ID[0] >= 27 and ID[0] <= 310]
        #     elif dataFilter == 'enst':
        #         list_IDs = [ID for ID in list_IDs if ID[0] >= 311 and ID[0] <= 373]
        #     elif dataFilter == 'bb':
        #         list_IDs = [ID for ID in list_IDs if ID[0] >= 374]
        
        return list_IDs

