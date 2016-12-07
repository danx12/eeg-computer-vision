#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:15:05 2016

@author: danielvillarreal
"""
import numpy as np
from PIL import Image
import imtools as it
import scipy as sp
import pylab as pl
from readStoredPickle import readStoredData,saveData


path = '/Volumes/DANFIT1/pat22_eeg_cv.p'
eeg = readStoredData(path)



#CZ,F3,C3 channels
data = eeg['data']
labels = eeg['labels']

#delete eeg to save memory
del eeg
#%%
nsamples,windowSize = data.shape
itemLabels = np.zeros([nsamples,windowSize])
for i in range(0,labels.shape[0]):
    itemLabels[i,:] = labels[i]

itemLabels = np.ravel(itemLabels)
event_range = []
i=0
while(i < len(itemLabels)):
    if(itemLabels[i] == 1):
        start = i
        for l in range(start,len(itemLabels)):
            if(itemLabels[l] == 0):
                end = l -1
                i += (l-start)
                event_range.append([start,end])
                break
    i+=1
event_range = np.asarray(event_range)
#%%
figure = pl.figure()
ax = figure.add_subplot(111)
ax.plot(np.ravel(data),'b')
y = [0,16000]
for i in range(0,22):
    ax.plot([event_range[i][0],event_range[i][0]],y,'r')
    ax.plot([event_range[i][1],event_range[i][1]],y,'g')


         