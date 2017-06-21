#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:23:43 2017

@author: viper
"""


matplotlib.use('pgf') # Force Matplotlib back-end

# Modules....
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from skimage import segmentation
from skimage.util import img_as_float
from skimage import color


plt.close('all') # Close all remaining figures


filename = 'I45.tiff'
im = io.imread(filename) # Open the image
im = img_as_float(im)
im=color.grey2rgb(im) 
plt.figure(1)
plt.imshow(im, cmap='gray')

labels=segmentation.felzenszwalb(im, scale=500, sigma=0.5, min_size=250)

plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))

