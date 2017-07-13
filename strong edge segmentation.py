#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:02:45 2017

@author: viper
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def orthoGrad(image, markers,z):
    nbRows, nbCols = image.shape
    x,y = markers
    x,y = int(x), int(y)
    neighboursList= np.asarray ([(x,max(0,y-1)), (max(x-1,0),y), (x,min(nbCols-1,y+1)), (min(x+1,nbRows-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(nbCols-1,y+1)), (min(x+1,nbRows-1),min(nbCols-1,y+1)), (min(x+1,nbRows-1),max(0,y-1))]) # L, T, R, B, TL, TR, BR, BL
    distance = np.asarray([abs(image[x,y]-image[int(a),int(b)]) for a,b in neighboursList])  
    # Get all possible new points to visit
    if distance.size is not 0:
        i = np.argmin(distance) # Direction du gradient
        a,b = i//4*4 + 1-i%2, i//4*4 + 3-i%2
        l = np.argmin([distance[a],distance[b]])
        markers = neighboursList[l]
    return markers

def getMarkers(image,n=2):
    """
    Interactive system to get initial markers. n the number of points.
    """
    plt.imshow(image, cmap='gray')
    markers = plt.ginput(n) # Get the two original seeds
    markers=np.asarray(markers) # Python list to Numpy array conversion
    x, y = markers.T
    for i in range(len(markers)): # Convert the seeds in order to be used with Numpy arrays
        x_,y_ = markers[i]
        markers[i]=[y_,x_]
    markers.astype(int) # Integer coordinates
    plt.close('all')
    return markers

image = io.imread('I0_0.tiff')

z = np.zeros(image.shape)
nx, ny = 10, 10
x = np.linspace(0, image.shape[0]-1, nx).astype(int)
y = np.linspace(0, image.shape[1]-1, ny).astype(int)
xv, yv = np.meshgrid(x, y)
w,h = xv.shape
for k in range(250):
    for i in range(w):
        for j in range(h):
            marker = [xv[i,j], yv[i,j]]
            a,b = marker
            z[a,b] +=1
            marker2=orthoGrad(image, marker,z)
            a,b = marker2
            xv[i,j], yv[i,j] = a, b
plt.imshow(color.label2rgb(z, image))