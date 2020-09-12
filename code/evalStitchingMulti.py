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

# YOUR CODE STARTS HERE

imCurrent = images[0]
for im in images[1:]:
    defaultH = np.array([[1,0,0], [0,1,0], [0,0,1]])
    imCurrent = warpImageWithMapping(imCurrent, im, defaultH)

cv2.imwrite(sys.argv[2], imCurrent)

cv2.imshow('Panorama', imCurrent)

cv2.waitKey()