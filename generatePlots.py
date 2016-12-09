#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:51:26 2016

@author: danielvillarreal
"""

import numpy as np
from PIL import Image
import imtools as it
import scipy as sp
import pylab as pl
from readStoredPickle import readStoredData,saveData
from sklearn.model_selection import cross_val_score,train_test_split

def fig2data (fig):

    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
    

def fig2img (fig):

    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    im = Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    im.load()
    bcg = Image.new('RGB',im.size,(255,255,255))
    bcg.paste(im,mask=im.split()[3])  
    return bcg.convert('L')
    
path = '/Volumes/DANFIT1/pat22_reduced.p'
eeg = readStoredData(path)


#Load differential channels Fp1-F3,F3-C3 and Fz-Cz
data = eeg['data']
labels = eeg['labels']

#%%
result = []
#pl.clf()

for i in range(0,data.shape[0]):
    print('Creating plot:',i)
    fig = pl.figure(figsize=(1.3,1.3),dpi=300,frameon=False)
    ax = pl.Axes(fig,[0.,0.,1.,1.])
    ax = fig.add_subplot(111)
    ax.plot(np.ravel(data[i,:]),linewidth=0.2)
    ax.set_ylim([-40,40])
    ax.axis('off')
    result.append(np.array(fig2img(fig)))
    pl.close(fig)
#%%
dataset = np.zeros([len(result),390*390])
for i in range(0,len(result)):
    dataset[i,:] = np.ravel(result[i])
    
#%%
#Split and shuffle into train and testing sets.
X_train,X_test,y_train,y_test = train_test_split(dataset,labels,test_size=0.3,random_state=41)

#build dictionary
eeg = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

#Save the dataset in the pickle format.
#saveData(eeg,'/Volumes/DANFIT1/eeg_dataset_split.p')