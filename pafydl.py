#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:53:21 2018

@author: grumiaux
"""

import pafy
import os

#%%
items_latina = []
with open('cwu/unlabeledDrumDataset/dataset_for_ismir2017/latin-songs_200_subset.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        if len(line.split('    ')) == 3:
            item = [line.split("    ")[0].split("   ")[0], line.split("    ")[0].split("   ")[1], line.split("    ")[1]]
#            link = line.split("    ")[1]
            items_latina.append(item)

items_rock = []
with open('cwu/unlabeledDrumDataset/dataset_for_ismir2017/hot-mainstream-rock-tracks_200_subset.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        if len(line.split('    ')) == 3:
            item = [line.split("    ")[0].split("   ")[0], line.split("    ")[0].split("   ")[1], line.split("    ")[1]]
#            link = line.split("    ")[1]
            items_rock.append(item)
            
items_pop = []
with open('cwu/unlabeledDrumDataset/dataset_for_ismir2017/pop-songs_200_subset.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        if len(line.split('    ')) == 3:
            item = [line.split("    ")[0].split("   ")[0], line.split("    ")[0].split("   ")[1], line.split("    ")[1]]
#            link = line.split("    ")[1]
            items_pop.append(item)
            
items_rnb = []
with open('cwu/unlabeledDrumDataset/dataset_for_ismir2017/r-b-hip-hop-songs_200_subset.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        if len(line.split('    ')) == 3:
            item = [line.split("    ")[0].split("   ")[0], line.split("    ")[0].split("   ")[1], line.split("    ")[1]]
#            link = line.split("    ")[1]
            items_rnb.append(item)

#%%
errors_latina = []
errors_pop = []
errors_rock = []
errors_rnb = []
for i, item in enumerate(items_latina):
    print(i, "===== ", item[0], item[1], " =====")
    try:
        video = pafy.new(item[2])
        bestaudio = video.getbestaudio()
        folder = './billboard/'    
        tempFilename = folder + 'billboard_latina_' + str(i) + '.' + str(bestaudio.extension)
        bestaudio.download(tempFilename)
    except:
        errors_latina.append(i)
print("########## LATINA SONGS DONE ##########")

for i, item in enumerate(items_rock):
    print(i, "===== ", item[0], item[1], " =====")
    try:
        video = pafy.new(item[2])
        bestaudio = video.getbestaudio()
        folder = './billboard/'    
        tempFilename = folder + 'billboard_rock_' + str(i) + '.' + str(bestaudio.extension)
        bestaudio.download(tempFilename)
    except:
        errors_pop.append(i)
print("########## ROCK SONGS DONE ##########")
    
for i, item in enumerate(items_pop):
    print(i, "===== ", item[0], item[1], " =====")
    try:
        video = pafy.new(item[2])
        bestaudio = video.getbestaudio()
        folder = './billboard/'    
        tempFilename = folder + 'billboard_pop_' + str(i) + '.' + str(bestaudio.extension)
        bestaudio.download(tempFilename)
    except:
        errors_rock.append(i)
print("########## POP SONGS DONE ##########")
    
for i, item in enumerate(items_rnb):
    try:
        print(i, "===== ", item[0], item[1], " =====")
        video = pafy.new(item[2])
        bestaudio = video.getbestaudio()
        folder = './billboard/'    
        tempFilename = folder + 'billboard_rnb_' + str(i) + '.' + str(bestaudio.extension)
        bestaudio.download(tempFilename)
    except:
        errors_rnb.append(i)
print("########## RNB SONGS DONE ##########")   

#%%
for i, item in enumerate(new_rnb_links):
#    print(i, "===== ", item[0], item[1], " =====")
    try:
        video = pafy.new(item)
        bestaudio = video.getbestaudio()
        folder = './billboard/'    
        tempFilename = folder + 'billboard_rnb_' + str(errors_rnb[i]) + '.' + str(bestaudio.extension)
        bestaudio.download(tempFilename)
        print(tempFilename)
    except:
#        errors_latina.append(i)
        print("error")
print("########## ROCK SONGS DONE ##########")
