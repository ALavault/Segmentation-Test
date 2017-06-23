#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:49:44 2017

@author: viper
"""

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
from PIL import Image
from  sklearn import cluster
from skimage.util import img_as_float

import PIL.Image as Image
filename = 'I0_0'
filename1 = filename+ '_s.png'
im = io.imread(filename1)
filename2 = filename+ '_w.png'
im2 = io.imread(filename2)
nbRows, nbCols = im.shape
for i in range(nbRows):
    for j in range(nbCols):
        if im[i,j] != 0:
            im[i,j] = 1
for i in range(nbRows):
    for j in range(nbCols):
        if im2[i,j] != 0:
            im2[i,j] = 2

imout = im + im2
im = Image.fromarray(imout)
im.save(filename+'_c.tiff')

im = io.imread(filename+'_c.tiff')
plt.imshow(im)
