#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:39:28 2018

@author: grumiaux
"""

import librosa
import os
import numpy as np

folder = "/users/grumiaux/Documents/stage/ENST-drums/"
audio_name = [f[:-4] for f in os.listdir(folder + "annotations/") if f.endswith(".txt")]

#%%
for audio in audio_name:
    x, sr = librosa.load(folder + "audio/mix66/" + audio + ".wav", sr=44100)
    mel = np.log(librosa.feature.melspectrogram(x, sr=sr, hop_length=441, fmin=20, fmax=20000, n_mels=84)+1)
    np.save(folder+"data/log_mel/mix66/" + audio, mel)
    print(audio + " ok")