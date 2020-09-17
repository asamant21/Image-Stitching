import os, sys
import cv2
import numpy as np
from utilsImageStitching import *
import matplotlib.pyplot as plt

# Load the left image and the right image.
imagePathLeft = '../data/uttower_left.jpg'
imagePathRight = '../data/uttower_right.jpg'

im_left = cv2.imread(imagePathLeft, cv2.IMREAD_GRAYSCALE)
im_right = cv2.imread(imagePathRight, cv2.IMREAD_GRAYSCALE)

# Implement the detectKeypoints() function in utilsImageStitching.py
# to detect feature points for both images. 
keypoints_left = detectKeypoints(im_left)
keypoints_right = detectKeypoints(im_right)

# Implement the computeDescriptors() function in utilsImageStitching.py
# to compute descriptors on keypoints
descriptors_left = computeDescriptors(im_left, keypoints_left)
descriptors_right = computeDescriptors(im_right, keypoints_right)

print(f"Getting Matches {len(descriptors_left)} with {len(descriptors_right)} key values")
# Implement the getMatches() function in utilsImageStitching.py
# to get matches
matches = getMatches(descriptors_left, descriptors_right)

print("Finished Matches")
drawMatches(im_left, im_right, matches, keypoints_left, keypoints_right)


# Implement the RANSAC() function in utilsImageStitching.py.
# Run RANSAC to estimate a homography mapping
H, numInliers = RANSAC(matches, keypoints_left, keypoints_right)

print("Finished RANSAC")
# Implement warpImageWithMapping() function in utilsImageStitching.py.
# Warp one image with the estimated homography mapping
# and composite the warpped image and another one.
panorama = warpImageWithMapping(im_left, im_right, H)
"""
plt.imshow(panorama, cmap='gray', vmin=0, vmax=255)

plt.show()"""