#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:21:20 2017

@author: viper

Test the following clustering methods :
    - K-means
    - DBSCAN
    - MeamShift
    
Note that every clustering process use an array, not a matrix
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
from  sklearn import cluster
from skimage import filters

import time


plt.close('all') # Close all remaining figures

filename = 'DoP.tiff'
im = io.imread(filename) # Open the image


# Matrix to array conversion
w, h = original_shape = tuple(im.shape)
d = 1
image_array = np.reshape(im, (w * h,1))


""" 
%    K-Means clustering method    %
"""

t=time.time()
Kmeans = cluster.KMeans(n_clusters=6, random_state=0).fit(image_array)
values = Kmeans.cluster_centers_.squeeze()
labels = Kmeans.predict(image_array)
image_kmeans = np.reshape(labels, (w, h))
n_clusters_ = len(np.unique(labels))

print("K-means clustering algorithm and post-treatment finished in : %0.3f s"% float(time.time()-t))
print('Estimated number of clusters: %d' % n_clusters_)



""" 
%    MeanShift clustering method    %
"""

t=time.time()
bandwidth = cluster.estimate_bandwidth(image_array, quantile=0.3, n_samples=500) # Estimate on n_sample sampled data
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True) # Much, much slower if bin_seeding=False (from ~1s to ~6min)
ms.fit(image_array)
labels = ms.predict(image_array)
n_clusters_ = len(np.unique(labels))
image_ms = np.reshape(labels, (w, h))

print("MeanShift clustering algorithm and post-treatment finished in : %0.3f s"% float(time.time()-t))
print('Estimated number of clusters: %d' % n_clusters_)



""" 
%    DBSCAN clustering method    %
"""

t=time.time()
db = cluster.DBSCAN(eps=0.5, min_samples=150).fit(image_array) # Execution time grows with eps, works badly with 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(np.unique(labels))
image_db = np.reshape(labels, (w, h))

print("DBSCAN clustering algorithm and post-treatment finished in : %0.3f s"% float(time.time()-t))
print('Estimated number of clusters: %d' % n_clusters_)



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


