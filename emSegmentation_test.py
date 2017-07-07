#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:57:07 2017

@author: viper

Description : Test of EM (Expectation-Maximization) segmentation method with Gaussian Mixture (refered to as GM)
"""
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage import color
from skimage import segmentation
from skimage import feature
from skimage import exposure
from skimage import filters
from sklearn import mixture
from skimage import util



def segmentEM(image, n_components=8, max_iter=100):    
    """
    Implements the EM segmentation
    Input :
        - image : image
    Output :
        - labelImage : thresholded matrix according to GM labels
    """
    if image.dtype !='uint8': # Force a conversion to 8bit image if the image is not of such type
        image = util.img_as_ubyte(image)
    hist, bins = exposure.histogram(image) # Histogram of the 8bit image
    hist = 1/float(image.size)*hist # Normalization -> pdf
    histogramArray = [hist, bins] # Histogram array for the GM
    histogramArray = np.asarray(histogramArray)
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=max_iter).fit(histogramArray.T)
    labels = gmm.predict(histogramArray.T) # Extract the labels by intensity
    labels = sortLabels(labels)

    labelImage = labelGM(image, labels) # Get the labelled image
    
    return labelImage, labels

def labelGM(image, labels):
    """
    Label an image given certain GM labels
    Inputs :
        - image : image
        - labels : labels of intensity obtained from a GM method
    Output :
        - labelImage : thresholded/labeled matrix according to GM labels
    """
    #Need 8 bits images
    w,h = image.shape
    labelImage = np.zeros(image.shape, dtype=np.int)
    mini = np.min(image)
    for i in range(w):
        for j in range(h):
            labelImage[i,j] = (labels[image[i,j]-mini])
    return labelImage    

def sortLabels(labels):
    """
    Sort the labels depending of their order of appearence in labels
    
    Inputs :
        - labels : labels of intensity obtained from a GM method
    Output :
        - labels : same vector as above
    """
    permutation = []
    for i in range(len(labels)):
        if labels[i] not in permutation:
            permutation.append(labels[i])
    k=0
    processed = [labels[0]]
    for i in range(len(labels)):
        if labels[i] not in processed:
            processed.append(labels[i])
            k+=1
            labels[i]=k
        else:
            idx = processed.index(labels[i])
            labels[i] = idx
    return labels

"""
#To test the function .....

plt.close('all')


im = io.imread('S1_0.tiff')
im = util.img_as_ubyte(im)


hist, bins = exposure.histogram(im) # Histogram of the 8bit image
hist = 1/float(im.size)*hist # Normalization -> pdf  

plt.plot(bins, hist, 'b-')

label, labels= segmentEM(im, n_components = 15, max_iter=100)

plt.figure(2)
plt.imshow(label, cmap = 'gray')
plt.figure(3)
plt.imshow(color.label2rgb(label, im))
"""
