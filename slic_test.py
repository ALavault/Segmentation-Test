#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:19:51 2017

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


filename = 'I135_0.tiff'
im = io.imread(filename) # Open the image
im = img_as_float(im)
im=color.grey2rgb(im) 
plt.figure(1)
plt.imshow(im, cmap='gray')

labels=segmentation.slic(im, n_segments=3, compactness=0.001, sigma=1) # sigma>0 => smoothed image

plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
plt.axis('off')


plt.savefig('Processed/Slic/'+filename)



