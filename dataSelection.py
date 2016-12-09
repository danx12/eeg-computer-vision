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
import random as rnd


path = '/Volumes/DANFIT1/pat22_eeg_multichan.p'
eeg = readStoredData(path)



#Load differential channels Fp1-F3,F3-C3 and Fz-Cz
data = eeg['data']
labels = eeg['labels']

#delete eeg to save memory
del eeg

nsamples,windowSize,chans, = data.shape
#%%

#Convert to uVolts
data = data.astype('float32') * 0.165

#%%
def plotFullDataset():
    #This is only needed to plot the whole dataset.

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
    
    figure = pl.figure()
    ax = figure.add_subplot(111)
    ax.plot(np.ravel(data[:,:,0]),'b')
    y = [np.min(data),np.max(data)]
    for i in range(0,22):
        ax.plot([event_range[i][0],event_range[i][0]],y,'r')
        ax.plot([event_range[i][1],event_range[i][1]],y,'g')
#%%
#Subsample dataset

positiveSamples = []
negativeSamples = []
for c in range(0,chans-1):
    for s in range(0,nsamples):
        if(labels[s] == 1):
            positiveSamples.append(data[s,:,c])
        else:
            negativeSamples.append(data[s,:,c])

positiveSamples = np.vstack(positiveSamples)
negativeSamples = np.vstack(negativeSamples)
del data
#Keep all positive, they are to rare to be wasted but randomly select the same number 
#of negative samples. In order to have balanced dataset.

nPositive = positiveSamples.shape[0]
idx = rnd.sample(range(0,negativeSamples.shape[0]),nPositive)
selectedNegative = negativeSamples[idx,:]

#Create new labels and combined samples.
ln = np.zeros(selectedNegative.shape[0])
lp = np.ones(positiveSamples.shape[0])
newLabels = np.hstack((ln,lp))
combinedSamples = np.vstack((selectedNegative,positiveSamples))


    
