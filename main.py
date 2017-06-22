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
from skimage import morphology
from skimage import color
from skimage import segmentation
from skimage import feature
from  sklearn import cluster
from skimage.util import img_as_float
import regiongrowing as rg
 




def watershed(im,markers,filename):
    plt.close('all') # Close all remaining figures
    plt.figure(1)
    x, y = markers.T # Extract the markers coordinates
    plt.plot(x, y, 'or', ms=4) # show where the markers are
    # Create a marker matrix

    markers_ws = np.zeros(im.shape, dtype=np.int)
    i=1
    for k in markers:
        x_,y_ = k
        markers_ws[x_.astype(np.int), y_.astype(np.int)] = i
        i+=1

    markers_ws = morphology.dilation(markers_ws, morphology.disk(3))
    canny = feature.canny(im, sigma = 3)
    canny=morphology.dilation(canny, morphology.disk(1))
    plt.imshow(canny, cmap='gray')
    labels_hat = segmentation.watershed(canny, markers_ws)
    from skimage import color
    color_labels = color.label2rgb(labels_hat, im)
    plt.imshow(segmentation.mark_boundaries(color_labels, labels_hat))
    plt.plot(x, y, 'or', ms=4)
    plt.axis('off')
    
    plt.savefig('Processed/Watershed/'+filename)
    plt.close('all')
def randomWalker(im,markers,filename):
    """
    Note 1 : ginput uses the coordinates of matplotlib which does not correspond to those of the matrix !
    To be more specific, the x axis for matplotlib.ginput represents the columns of the matrix when the y axis represents 
    the lines of the matrix.
    """
    x, y = markers.T
    plt.imshow(im, cmap='gray')
    plt.plot(x, y, 'or', ms=4) # show where the markers are
    # Array of markers
    markers_rw = np.zeros(im.shape, dtype=np.int)
    i=1
    for k in markers:
        x,y = k
        markers_rw[x.astype(np.int), y.astype(np.int)] = i # See Note 1
        i+=1     
    #markers_rw = morphology.dilation(markers_rw, morphology.disk(25)) # Give more "weight" to the markers 
    plt.imshow(markers_rw) 
    from skimage import segmentation
    labels_rw = segmentation.random_walker(im, markers_rw, beta=2500, mode='cg_mg')
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels_rw, im), labels_rw))
    plt.plot(y, x, 'ok', ms=2)
    plt.savefig('Processed/Random Walker/'+filename)
    plt.close('all')
def clustering(im, filename):
    # Matrix to array conversion
    w, h  = tuple(im.shape)
    image_array = np.reshape(im, (w * h,1))
    """ 
    %    K-Means clustering method    %
    """
    Kmeans = cluster.KMeans(n_clusters=6, random_state=0).fit(image_array)
    labels = Kmeans.predict(image_array)
    image_kmeans = np.reshape(labels, (w, h))
    n_clusters_ = len(np.unique(labels))  
    #print('Estimated number of clusters: %d' % n_clusters_)
    """ 
    %    MeanShift clustering method    %
    """
    bandwidth = cluster.estimate_bandwidth(image_array, quantile=0.3, n_samples=500) # Estimate on n_sample sampled data
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True) # Much, much slower if bin_seeding=False (from ~1s to ~6min)
    ms.fit(image_array)
    labels = ms.predict(image_array)
    n_clusters_ = len(np.unique(labels))
    image_ms = np.reshape(labels, (w, h))
    #print('Estimated number of clusters: %d' % n_clusters_)   
    """ 
    %    DBSCAN clustering method    %
    """
    db = cluster.DBSCAN(eps=0.5, min_samples=150).fit(image_array) # Execution time grows with eps, works badly with 
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(np.unique(labels))
    image_db = np.reshape(labels, (w, h))
    #print('Estimated number of clusters: %d' % n_clusters_)        
    """
    Plotting the results
    """
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(im, cmap='gray')
    axarr[0,0].set_title('Original Image')
    axarr[0,0].axis('off')
    axarr[0,1].imshow(image_kmeans, cmap='plasma')
    axarr[0,1].set_title('K-Means')
    axarr[0,1].axis('off')
    axarr[1,0].imshow(image_ms, cmap='plasma')
    axarr[1,0].set_title('MeanShift')
    axarr[1,0].axis('off')
    axarr[1,1].imshow(image_db, cmap='plasma')
    axarr[1,1].set_title('DBSCAN')
    axarr[1,1].axis('off')
    plt.savefig('Processed/Clustering/'+filename,dpi = 96*len(axarr))
    plt.close('all') 
def felzenszwalb(im, filename):
    plt.close('all') # Close all remaining figures
    im = img_as_float(im)
    im=color.grey2rgb(im) 
    plt.figure(1)
    plt.imshow(im, cmap='gray')
    labels=segmentation.felzenszwalb(im, scale=800, sigma=0.5, min_size=250) 
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
    plt.savefig('Processed/Felzenszwalb/'+filename)
    
def slic(im, filename):
    plt.close('all') # Close all remaining figures
    im = img_as_float(im)
    labels=segmentation.slic(im, n_segments=3, compactness=0.001, sigma=1) # sigma > 0 => smoothed image
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
    plt.axis('off')    
    plt.savefig('Processed/Slic/'+filename)

def quickshift(im, filename):
    plt.close('all') # Close all remaining figures
    im = img_as_float(im)
    canny = feature.canny(im, sigma=6)
    plt.figure(1)
    plt.imshow(canny, cmap='gray')
    mu=2
    im-= mu*canny
    im=color.gray2rgb(im)
    labels=segmentation.quickshift(im, kernel_size=5, max_dist=600, ratio=0.9,sigma=0)
    #print('Quickshift number of segments: {}'.format(len(np.unique(labels))))
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
    plt.savefig('Processed/Quickshift/'+filename)

    
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, im), labels))
def regionGrowing(image, seeds, pixelThreshold, regionThreshold, filename):
    plt.close('all') # Close all remaining figures
    y, x = seeds.T

    labels = rg.regionGrowing(image, seeds, pixelThreshold, regionThreshold)
    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image), labels))
    plt.plot(x, y, 'or', ms=3)
    plt.savefig('Processed/Region Growing/'+filename)



    
def main():
    
    """
    Description : process every relevant file (i.e. images) in the current work directory
    Need an existing 'Processed' folder with two subfolders in it ('Watershed' and 'Random Walker')
    """
    plt.close('all') # Close all remaining figures
    n= 3
    pTh= 5000
    rTh = 3250
    isFirstIteration=True
    fileList = os.listdir(os.getcwd())
    for filename in fileList:
        try:
            print('In Process : '+filename)
            if isFirstIteration:

                im = io.imread(filename) # Open the image
                print(type(im[0,0]))

                plt.figure(1)
                plt.imshow(im, cmap='gray')
                print('Choose '+str(n)+ ' points for segmentation in this order : Sky, water, others')
                markers = plt.ginput(n) # n points to choose as markers/seeds
                markers=np.asarray(markers) # Convert a Python list to a Numpy array
                seeds=markers
                isFirstIteration = False
                for i in range(len(seeds)):
                    x_,y_ = seeds[i]
                    seeds[i]=[y_,x_]
                markers.astype(int)
                seeds.astype(int)
                watershed(im,markers,filename)                
                randomWalker(im,markers,filename)
                clustering(im, filename)
                felzenszwalb(im, filename)
                slic(im, filename)
                quickshift(im, filename)
                if (im[0,0].dtype == 'uint8'):
                    regionGrowing(im, seeds, pTh/256, rTh/256, filename)
                else:
                    regionGrowing(im, seeds, pTh, rTh, filename)

                print('Done')
            else:
                im = io.imread(filename) # Open the image
                #markers=np.asarray(markers) # Convert a Python list to a Numpy array
                watershed(im,markers,filename)
                randomWalker(im,markers,filename)
                clustering(im, filename)
                felzenszwalb(im, filename)
                slic(im, filename)
                quickshift(im, filename)
                if (im[0,0].dtype == 'uint8'):
                    regionGrowing(im, seeds, pTh/256, rTh/256, filename)
                else:
                    regionGrowing(im, seeds, pTh, rTh, filename)

        except IOError:
            print(filename+' : Not a file or not an image (IOError)')
    plt.close('all')
    print('Done !')




        
        
if __name__ == "__main__":
    main()