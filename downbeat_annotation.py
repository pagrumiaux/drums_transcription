#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:06:14 2018

@author: grumiaux
"""

import madmom
import os

bb_folder = 'rbma_13/'
act_proc = madmom.features.RNNDownBeatProcessor()
dbn_proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

l = [e for e in os.listdir(bb_folder + 'audio/') if e.endswith('.wav')]
for i, f in enumerate(l):
    print(str(i) + ". " + f + " being processed...")
    act = act_proc(bb_folder + 'audio/' + f)
    result = dbn_proc(act)
    
    lines = []
    for r in result:
        lines.append(str(r[0]) + " " + str(int(r[1])) + '\n')
    with open(bb_folder + 'annotations/beats_madmom/' + f[:-4] + ".txt", "w") as fil:
        fil.writelines(lines)    
    
    print("---> " + f + " done.")