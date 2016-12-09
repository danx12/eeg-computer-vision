#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:37:25 2016

@author: danielvillarreal
"""

import numpy as np
from PIL import Image
import imtools as it
import scipy as sp
import pylab as pl
from readStoredPickle import readStoredData,saveData
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score


def sumInt(im,i,j,w,h):
    return im[i+h-1,j+w-1] - im[i,j+w-1] - im[i+h-1,j] + im[i,j]

def computeHaarRectangle(im,i,j,w,h,rectangle):
    haar_feature = 0;
#    Vertical edge
    if rectangle == 0:
        bright = sumInt(im,i,j,w/2,h)
#        bright = im[i+h-1,j+w/2-1] - im[i-1,j+w/2-1] - im[i+h-1,j-1] + im[i-1,j-1]
#        dark = im[i+h-1,j+w-1] - im[i-1,j+w-1] - im[i+h-1,j+w/2-1] + im[i-1,j+w/2-1]
        dark = sumInt(im,i,j+w/2,w/2,h)

        haar_feature = bright-dark
#     Horizontal edge   
    elif rectangle == 1:
#        bright = im[i+h/2-1,j+w-1] - im[i-1,j+w-1] - im[i+h/2-1,j-1] + im[i-1,j-1]
#        dark = im[i+h-1,j+w-1] - im[i+h/2-1,j+w-1] - im[i+h-1,j-1] + im[i+h/2-1,j-1]
        bright = sumInt(im,i,j,w,h/2)
        dark = sumInt(im,i+h/2,j,w,h/2)
        haar_feature = bright-dark
#   Vertical line
    elif rectangle == 2:
        bright1 = sumInt(im,i,j,w/3,h)
        dark = sumInt(im,i,j+w/3,w/3,h)
        bright2 = sumInt(im,i,j+2*(w/3),w/3,h)
        haar_feature = bright1+bright2-dark
#   Horizontal line
    elif rectangle == 3:
        bright1 = sumInt(im,i,j,w,h/3)
        dark = sumInt(im,i+h/3,j,w,h/3)
        bright2 = sumInt(im,i+2*(h/3),j,w,h/3)
        haar_feature = bright1+bright2-dark
#    Diagonal rectangle
    elif rectangle == 4:
        bright = sumInt(im,i+h/2,j,w/2,h/2) + sumInt(im,i,j+w/2,w/2,h/2)
        dark = sumInt(im,i,j,w/2,h/2) + sumInt(im,i+h/2,j+w/2,w/2,h/2)
        haar_feature = bright-dark
        
    return haar_feature
    
def computeBestRectangle(RegionSet,labels,i,j,w,h):
    nSamples= RegionSet.shape[0]
    results = np.zeros(5)
    for f in range(0,5):
        vect = np.zeros(nSamples)
        for s in range(0,nSamples):
            vect[i] = computeHaarRectangle(RegionSet[s,:,:],i,j,w,h,f)
        clf = DecisionTreeClassifier(random_state=0)
#        clf = svm.SVC(kernel='linear', C=1)
#        score = cross_val_score(clf, vect.reshape(-1, 1), np.ravel(labels), cv=3)
        clf.fit(vect.reshape(-1, 1),labels)
        score = clf.score(vect.reshape(-1, 1),np.ravel(labels))
        results[f] = np.mean(score)
    bestRectangle = np.argmax(results)
    return results[bestRectangle],bestRectangle

def computeBestSubRegion(IntegralRegionSet,labels,rsize):
    bestFeats = []
    subregionsToCompute = [30,15,10]
    nSubWindows = IntegralRegionSet.shape[1]
    for reg in range(0,nSubWindows):
        print("Computing region: %d of %d" % (reg,nSubWindows))
#        for sub in range(0,len(subregionsToCompute)):
        best_rect = 0;
        best_params = [0,0,0,0]
        best_score = -100
        for size in range(4,rsize):
            print("Computing size: %d of %d" % (size,rsize))
            for i in range(0,rsize-size):
                for j in range(0,rsize-size):
                    
                    
                    score,rect = computeBestRectangle(IntegralRegionSet[:,reg,:,:],labels,i,j,
                                                 size,size)
#            for i in np.linspace(0,rsize,rsize/subregionsToCompute[sub]+1)[:-1]:
#                score,rect = computeBestRectangle(IntegralRegionSet[:,reg,:,:],labels,i,i,
#                                                 subregionsToCompute[sub],subregionsToCompute[sub])
                    if(score >= best_score):
                        best_score = score
                        best_rect = rect
                        best_params = [i,j,size]
        bestFeats.append({'reg':reg,'rect':best_rect,'best_params':best_params,'best_score':best_score})
    
    return bestFeats
def getIntegralImage(dataset):
    nsamples = dataset.shape[0];
    intset = np.zeros(dataset.shape)
    for i in range(0,nsamples):
        intset[i,:,:] =  np.cumsum(np.cumsum(dataset[i,:,:],axis=1),axis=0)
    return intset

def createRegions(im,rsize):
    result = np.zeros([169,30,30])
    i = 0
    for m in np.arange(0,390,rsize):
        for n in np.arange(0,390,rsize):
            result[i,:] =(im[m:m+rsize,n:n+rsize])
            i+=1
    return result
    
def datasetRegions(dataset,rsize):
    nsamples = dataset.shape[0]
    results = np.zeros([nsamples,169,rsize,rsize])
    for i in range(0,nsamples):
        results[i,:] = createRegions(dataset[i,:],rsize)
    return results
        

        
path = '/Volumes/DANFIT1/eeg_dataset_split.p'
eeg = readStoredData(path)

#Load the plots for a reduced dataset.
X_train = eeg['X_train']
y_train = eeg['y_train']

X_train = np.reshape(X_train,[X_train.shape[0],390,390])

X_train = getIntegralImage(X_train)

X_train_regions = datasetRegions(X_train,30)

results = computeBestSubRegion(X_train_regions,y_train,30)






