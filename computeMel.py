#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:39:28 2018

@author: grumiaux
"""

import librosa
import os
import numpy as np

folder = "/users/grumiaux/Documents/stage/billboard/"
audio_name = [f[:-4] for f in os.listdir(folder + "audio/") if f.endswith(".mp3")]

#%%
for audio in audio_name:
    x, sr = librosa.load(folder + "audio/" + audio + ".mp3", sr=44100)
    mel = np.log(librosa.feature.melspectrogram(x, sr=sr, hop_length=441, fmin=20, fmax=20000, n_mels=84)+1)
    np.save(folder+"data/log_mel/" + audio, mel)
    print(audio + " ok")