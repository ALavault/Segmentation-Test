#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:12:37 2017

@author: viper


"""


# Modules....
import os
import matplotlib
matplotlib.use('pgf')
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from skimage import morphology
from skimage import color
from skimage import segmentation



def watershed(im,markers,filename):
    
    """
    Inputs :
        - im : numpy array, represents an image (can be multidimensionnal)
        - markers : Numpy array of markers/seeds
        - filename : name used to save the processed image
        
    Output : Nothing (Save the processed image)
    
    Description : implements the Watershed segmentation method

    """
    
    plt.close('all') # Close all remaining figures
    plt.figure(1)
    plt.imshow(im, cmap='gray')    
    markers=np.asarray(markers)
    x, y = markers.T # Extract the markers coordinates
    
    # Create a marker matrix   
    markers = np.zeros(im.shape, dtype=np.int)
    markers[x.astype(np.int), y.astype(np.int)] = np.arange(len(x)) + 1
    markers = morphology.dilation(markers, morphology.disk(7))
    
    # Watershed Segmentation Algorithm, as seen at http://emmanuelle.github.io/a-tutorial-on-segmentation.html
    
    # Black tophat transformation 
    hat = ndimage.black_tophat(im, 7)
    mu=0.3 # Coefficient 
    # Combine with the original image (a type conversion is compulsory)
    hat =hat.astype('float64')- mu * im.astype('float64')
    plt.figure(2)
    plt.imshow(hat, cmap='gray')
    labels_hat = segmentation.watershed(hat, markers)
    color_labels = color.label2rgb(labels_hat, im)
    plt.figure(3)
    plt.imshow(segmentation.mark_boundaries(color_labels, labels_hat))
    plt.plot(x, y, 'or', ms=4)
    plt.savefig('Processed/Watershed/'+filename) # Save the processed image
    plt.close('all')

def randomWalker(im,markers,filename):
    """
    Inputs :
        - im : numpy array, represents an image (can be multidimensionnal)
        - markers : Numpy array of markers/seeds
        - filename : name used to save the processed image
        
    Output : Nothing (Save the processed image)
    
    Description : implements the Random Walker segmentation method
    """    
    x, y = markers.T
    plt.imshow(im, cmap='gray')
    plt.plot(x, y, 'or', ms=4) # show where the markers are
    
    # Array of markers
    markers_rw = np.zeros(im.shape, dtype=np.int)
    i=1
    for k in markers:
        x,y = k
        markers_rw[x.astype(np.int), y.astype(np.int)] = i
        i+=1        
    markers_rw = morphology.dilation(markers_rw, morphology.disk(25))
    
    plt.imshow(markers_rw)
    
    labels_rw = segmentation.random_walker(im, markers_rw, beta=2500, mode='cg_mg')
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels_rw, im), labels_rw))
    plt.plot(y, x, 'ok', ms=2)
    plt.savefig('Processed/Random Walker/'+filename)
    plt.close('all')


def main():
    
    """
    Description : process every relevant file (i.e. images) in the current work directory
    Need an existing 'Processed' folder with two subfolders in it ('Watershed' and 'Random Walker')
    """
    n=2
    isFirstIteration=True
    fileList = os.listdir(os.getcwd())
    for filename in fileList:
        try:
            print('In Process : '+filename)
            if isFirstIteration:
                im = io.imread(filename) # Open the image
                plt.figure(1)
                plt.imshow(im, cmap='gray')
                print('Choose '+str(n)+ ' points for segmentation')
                markers = plt.ginput(n) # n points to choose as markers/seeds
                markers=np.asarray(markers) # Convert a Python list to a Numpy array
                isFirstIteration=False
                watershed(im,markers,filename)                
                randomWalker(im,markers,filename)
                print('Done')
            else:
                im = io.imread(filename) # Open the image
                markers=np.asarray(markers) # Convert a Python list to a Numpy array
                watershed(im,markers,filename)
                randomWalker(im,markers,filename)
        except IOError as e:
            print(filename+' : Not a file or not an image (IOError)')
    plt.close('all')
    print('Done !')




        
        
if __name__ == "__main__":
    main()