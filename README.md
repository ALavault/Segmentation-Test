# Segmentation tests

## Global presentation
The following methods have been implemented and tested :
- Watershed (with 2 and 3 seeds)
- Random Walker (with 2 and 3 seeds)
- Felzenswalb (no seeds required)
- Quickshift (no seeds required)
- Slic (no seeds required, number of segments required)
- Clustering methods (no seeds required)
	* K-means (number of clusters required)
	* MeanShift
	* DBSCAN
- Region Growing (with three seeds and naive algorithm)
- Expectation–maximization algorithm

## Details and remarks
Watershed method :
- Uses segmentation from skimage. Uses a gradient of the image as heightmap for flooding.
Random Walker :
- Uses segmentation from skimage.
Felzenswalb :
- Uses segmentation from skimage.
Quickshift :
- Uses segmentation from skimage. 
Slic :
- Uses segmentation from skimage. Number of segments can be changed.
Clustering methods :
- Use scikit-learn. A number of clusters has to be given to the k-means method. DBSCAN method can rapidly give a lot of clusters (>100).
Region Growing :
- Graph search like -> from each seeds, find all neighbours which are within a threshold, "seed" them and visit the next point to be visited.
- Not optimized. 
- Has an option to force the seeding of unseeded pixels (because of thresholds too tight for instance)
- Has an option to sort to be visited pixels but does not give any results.
Expectation–maximization :
- Uses scikit-learn to implement an expectation-maximization algorithm based on gaussian mixture

