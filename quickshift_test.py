#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:27:13 2017

@author: viper
"""


matplotlib.use('pgf') # Force Matplotlib back-end

# Modules....
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import segmentation
from skimage.util import img_as_float
from skimage import color
from skimage import feature

plt.close('all') # Close all remaining figures


filename = 'DoP.tiff'
im = io.imread(filename) # Open the image
im = img_as_float(im)

canny = feature.canny(im, sigma=6)

plt.figure(1)
plt.imshow(canny, cmap='gray')
mu=2
im-= mu*canny
im2=im
im=color.gray2rgb(im)
plt.figure(3)


labels=segmentation.quickshift(im, kernel_size=5, max_dist=600, ratio=0.9,sigma=0)
print('Quickshift number of segments: {}'.format(len(np.unique(labels))))

plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))

