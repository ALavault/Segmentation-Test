#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:48:40el 2017

@author: viper
"""

import main

# Modules....
import os
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage import color
from skimage import segmentation
from skimage import feature
from skimage import filters

from  sklearn import cluster
from sklearn import metrics
from skimage.util import img_as_float
import regiongrowing as rg

import PIL.Image as Image

def getConfusionMatrix(groundTruth, prediction):
    confusionMatrix = np.zeros((3,3))
    nbRows, nbCols = groundTruth.shape
    for i in range(nbRows):
        for j in range(nbCols):
            if groundTruth[i,j]==prediction[i,j]%3:
                confusionMatrix[groundTruth[i,j], groundTruth[i,j]]+=1
            else:
                confusionMatrix[groundTruth[i,j], prediction[i,j]%3]+=1
    print confusionMatrix
    bgAcc=(confusionMatrix[0,0]+np.sum(confusionMatrix[[1,2],[1,2]]))/prediction.size
    minor = confusionMatrix[:,[0,2]]
    skyAcc=(confusionMatrix[1,1]+np.sum(minor[[0,2],:]))/prediction.size
    minor = confusionMatrix[:,[0,1]]
    waterAcc=(confusionMatrix[2,2]+np.sum(minor[[0,1],:]))/prediction.size
    
    
    
    print('Background Accuracy = '+str(bgAcc*100)+'%\nSky Accuracy ='+str(skyAcc*100) + '%\nWater Accuracy ='+str(waterAcc*100)  +'%')
    return confusionMatrix


plt.close('all')
groundTruthfname = 'Base/S1_0_c.tiff'
gt = io.imread(groundTruthfname)
imagefname = 'S1_0.tiff'
im = io.imread(imagefname)
n=3
plt.figure(1)
plt.imshow(im, cmap='gray')
print('Choose '+str(n)+ ' points for segmentation in this order : Sky, water, others')
markers = plt.ginput(n) # n points to choose as markers/seeds
print('Init done')
markers=np.asarray(markers) # Convert a Python list to a Numpy array
seeds=markers
isFirstIteration = False
for i in range(len(seeds)):
    x_,y_ = seeds[i]
#    seeds[i]=[y_,x_]
markers.astype(int)
seeds.astype(int)
i = im.max()-im.min()
pTh= 5000
rTh = 3150
matList =[]



#grad = filters.rank.gradient(im, morphology.disk(5))
#labels =  main.regionGrowing(im, seeds, pTh, rTh, 'image.png', matList)
#labels =  main.randomWalker(im, seeds, 'image.png')
labels =  main.watershed(im, seeds, 'image.png')          
            



w, h = original_shape = tuple(im.shape)
gt_array = np.reshape(gt, (w * h,1))
for i in range(w):
    for j in range(h):
        if labels[i,j]>=3:
            labels[i,j]=0
labels_array = np.reshape(labels, (w * h,1))

cm = metrics.confusion_matrix(gt_array, labels_array)
acc = metrics.classification_report(gt_array, labels_array, target_names=['Background', 'Sky', 'Water'])
i = im.max()-im.min()
print(i)
print(imagefname)
print(acc)

plt.figure(1)
plt.imshow(color.label2rgb(labels, im))
plt.figure(2)
plt.imshow(color.label2rgb(gt, im))
plt.figure(3)
plt.imshow(hm)