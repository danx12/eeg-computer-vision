#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:43:49 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
import sklearn as sk
import scipy as sp
import glob
import pickle as pickle 
from filter_lowpass import lowpassFilter
from readStoredPickle import readStoredData,saveData

eventInfo = np.genfromtxt('events.csv',delimiter=',',dtype='int')
numEvents = eventInfo.shape[0]

data = []
labels = []

dirpath = '/Volumes/DANFIT1/HifoCap/surf30/pat_22602/adm_226102/rec_22600102'
for i,path in enumerate(glob.glob('%s/22600102_*.data' % dirpath)):
    print("Processing batch: " + str(i))
    batch = np.fromfile(path,dtype='int16',count=-1)
    rows = len(batch)/29
    batch = (batch.reshape([rows,29]).astype('int16') * 1)[:,[0,2,4,16,17]]
    data.append(batch)
    lb = np.zeros([rows,1])
    for f,start,end in eventInfo:
        if(f == i):
            lb[start:end+1,:] = 1
    labels.append(lb)
        
data = np.concatenate(data,axis=0)
labels = np.concatenate(labels,axis=0)

def windowsOfData(data,wsize,samplingrate):
    rows,channels = data.shape
    samplesPerWindow = rows / float(samplingrate)
    samplesPerWindow /= wsize
    windows = np.zeros([int(samplesPerWindow),int(samplingrate),channels])
    windows = np.array(np.split(data,samplesPerWindow))
    return windows,samplesPerWindow

#convert data and labels into windows 
data,samplesPerWindow = windowsOfData(data,1,256)
labels = np.sum(np.split(labels,samplesPerWindow),axis=1)
labels = np.where(labels > 0,1,0).astype('uint16')

#%%
dims = data.shape
data = data.astype('int32')

bi_data = np.zeros([dims[0],dims[1],3],dtype='int32')
bi_data[:,:,0] = data[:,:,0] - data[:,:,1]
bi_data[:,:,1] = data[:,:,1] - data[:,:,2]
bi_data[:,:,2] = data[:,:,3] - data[:,:,4]

#del data
#convertToVoltsFactor = float(0.165000)

#CZ,F3,C3 channels

#bi_data = bi_data * convertToVoltsFactor

#%%
#data = lowpassFilter(data,2)
#data = data - np.mean(data)
#data = (data + abs(np.min(data)))

