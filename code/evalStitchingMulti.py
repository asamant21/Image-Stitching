import os, sys
import cv2
import numpy as np
from utilsImageStitching import *

imagePath = sys.argv[1]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE))

# Build your strategy for multi-image stitching. 
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the 
# number of inliers to each transformation, use this information to determine which 
# pair of images should be merged first (and of these, which one should be the "source" 
# and which the "destination"), merge this pair, and proceed recursively.

keypoints = []
descriptors = []
for i in range(len(images)):
	keypoints.append(detectKeypoints(images[i]))
	descriptors.append(computeDescriptors(images[i], keypoints[i]))

num_inliers_matrix = []
for i in range(len(images)):
	num_inliers_list = [0] * i
	for j in range(i, len(images)):
		matches = getMatches(descriptors[i], descriptors[j])
		_, numInliers = RANSAC(matches, keypoints[i], keypoints[j])
		num_inliers_list.append(numInliers)

	num_inliers_matrix.append(num_inliers_list)


for i in range(len(num_inliers_matrix)):
	idx = np.argpartition(num_inliers_matrix[i], -2)[-2:]
	indices = idx[np.argsort((-num_inliers_matrix[i])[idx])]

	if num_inliers_matrix[i][indices[0]] / num_inliers_matrix[i][indices[1]]



# compute num inliers between all pairs of images
# start w *rightmost* image
# determine by:
# - getting num inliers between all images
# - image w only one "friend" will be either leftmost or rightmost
# - for these two images: determine if it's on "left" or "right" by 
#   comparing *average col values* of its *keypoints that got matched*

imCurrent = images[0]
for im in images[1:]:
    
    imCurrent = warpImageWithMapping(imCurrent, im, defaultH)

cv2.imwrite(sys.argv[2], imCurrent)

cv2.imshow('Panorama', imCurrent)

cv2.waitKey()