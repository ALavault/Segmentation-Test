#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:51:57 2017

@author: viper

Random walker segmentation test
"""

matplotlib.use('pgf')

plt.close('all')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from skimage import morphology
from skimage import color
from skimage import filters

matplotlib.use('pgf')


filename = 'S0_0.tiff'
im = io.imread(filename)
#im = filters.gaussian(im,sigma=1) # filtering : for smoother boundaries
plt.figure(1)
plt.imshow(im, cmap='gray')


markers = plt.ginput(n=3) # n points to choose
"""
Note 1 : ginput uses the coordinates of matplotlib which does not correspond to those of the matrix !
To be more specific, the x axis for matplotlib.ginput represents the columns of the matrix when the y axis represents 
the lines of the matrix.

"""
markers=np.asarray(markers) # Python list to Numpy array conversion

x, y = markers.T
plt.imshow(im, cmap='gray')
plt.plot(x, y, 'or', ms=4) # show where the markers are

# Array of markers
markers_rw = np.zeros(im.shape, dtype=np.int)
print(im.shape)
i=1
for k in markers:
    x,y = k
    markers_rw[y.astype(np.int), x.astype(np.int)] = i # See Note 1
    i+=1
    
#markers_rw = morphology.dilation(markers_rw, morphology.disk(25)) # Give more "weight" to the markers

plt.imshow(markers_rw)

from skimage import segmentation
labels_rw = segmentation.random_walker(im, markers_rw, beta=2500, mode='cg_mg')
plt.figure(1)

plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels_rw, im), labels_rw))

plt.plot(y, x, 'ok', ms=2)
plt.savefig('Processed/Random Walker/'+filename)


