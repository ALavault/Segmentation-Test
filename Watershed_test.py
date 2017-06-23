#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:51:57 2017

@author: viper

Watershed segmentation test
"""


matplotlib.use('pgf') # Force Matplotlib back-end

# Modules....
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from skimage import morphology
from skimage import segmentation
from skimage import filters
from skimage import feature


plt.close('all') # Close all remaining figures


filename = 'DoP_0.tiff'
im = io.imread(filename) # Open the image
plt.figure(1)
plt.imshow(im, cmap='gray')
markers = plt.ginput(n=3) # n points to choose as markers/seeds
markers=np.asarray(markers) # Convert a Python list to a Numpy array

x, y = markers.T # Extract the markers coordinates
plt.imshow(im, cmap='gray')
plt.plot(x, y, 'or', ms=4) # show where the markers are

# Create a marker matrix

markers_ws = np.zeros(im.shape, dtype=np.int)
i=1
for k in markers:
    x_,y_ = k
    markers_ws[y_.astype(np.int), x_.astype(np.int)] = i
    i+=1
markers_ws = morphology.dilation(markers_ws, morphology.disk(3))


# Watershed, as seen at http://emmanuelle.github.io/a-tutorial-on-segmentation.html
"""
# Black tophat transformation 
hat = ndimage.white_tophat(im, 7)
# Combine with the original image (a type conversion is compulsory) to create the heightmap for the algorithm
mu=1.8 # Coefficient 
hat =hat.astype('float64')- mu * im.astype('float64')
"""

hat = feature.canny(im, sigma = 1)
hat=morphology.dilation(hat, morphology.disk(1))
plt.figure(2)
plt.imshow(hat, cmap='gray')
labels_hat = segmentation.watershed(hat, markers_ws)
from skimage import color
color_labels = color.label2rgb(labels_hat, im)
plt.figure(3)
plt.imshow(segmentation.mark_boundaries(color_labels, labels_hat))
plt.plot(x, y, 'or', ms=4)
plt.axis('off')

plt.savefig('Processed/Watershed/'+filename)



