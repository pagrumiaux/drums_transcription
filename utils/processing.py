# -*- coding: utf-8 -*-
"""
Created on Mon Mar  15 21:16:02 2021

@author: PA
"""

import librosa
import os
import numpy as np
import madmom

def compute_mel_spectrograms(input_folder,
                             output_folder,
                             sr=44100, 
                             hop_length=441, 
                             fmin=20, 
                             fmax=20000,
                             n_mels = 84):
    """Scan through a folder with .mp3 and compute the mel spectrograms
    for each audio file.

    Parameters
    ----------
    input_folder : str
        Path of the folder containing the .mp3 files.
    output_folder : str
        Path of the folder to save the mel spectrograms.
    sr : int, optional
        Sampling frequency, by default 44100
    hop_length : int, optional
        Hop length between 2 frames, by default 441
    fmin : int, optional
        Frequency of the low mel filter, by default 20
    fmax : int, optional
        Frequency of the high mel filter, by default 20000
    n_mels : int, optional
        Number of mel filters, by default 84
    """                             
    audio_filenames = [f[:-4] for f in os.listdir(input_folder) \
                       if f.endswith('.mp3')]
    for audio in audio_filenames:
        x, sr = librosa.load(input_folder + audio + '.mp3', sr = sr)
        mel_spectrogram = np.log(librosa.feature.melspectrogram(x, sr=sr, \
            hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)+1)
        np.save(output_folder + audio, mel)
    print(f'All .mp3 files in folder {input_folder} computed into mel \
          spectrograms in folder {output_folder}')

def annotation_downbeat(input_folder, annotation_folder):
    """Annotate downbeat information for all .wav files in specified folder.

    Parameters
    ----------
    input_folder : str
        Folder containing the .wav files.
    annotation_folder : str
        Folder to save the annotations in.
    """    
    act_processor = madmom.features.RNNDownBeatProcessor()
    dbn_processor = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4],
                                                                 fps=100)
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    for iFile, audio_file in enumerate(audio_files):
        print(f'{str(iFile)}. {audio_file} being processed...')
        act = act_proc(input_folder + audio_file)
        result = dbn_proc(act)

    lines = []
    for r in result:
        lines.append(f'{str(r[0])} {str(int(r[1]))}\n')
    
    with open(f'{annotation_folder}{audio_file[:-4]}.txt', 'w') as annotation_file:
        annotation_file.writelines(lines)

    print(f'{audio_file} annotated.')