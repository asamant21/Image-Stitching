import os, sys
import cv2
import numpy as np
from utilsImageStitching import *
import matplotlib.pyplot as plt

#imagePath = sys.argv[1]
imagePath = '../data/image_sets/ledge/'
images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE))

# Build your strategy for multi-image stitching. 
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the 
# number of inliers to each transformation, use this information to determine which 
# pair of images should be merged first (and of these, which one should be the "source" 
# and which the "destination"), merge this pair, and proceed recursively.
def find_and_stitch_closest_image(current_image, images, keypoints, descriptors):
    current_image_keypoints = detectKeypoints(current_image)
    current_image_descriptors = computeDescriptors(current_image, current_image_keypoints)
    max_H = np.empty(shape=(3,3))
    maxInliers = 0
    max_matches = (np.empty(shape=(0)), np.empty(shape=(0)))
    max_index = -1
    is_left = True

    MATCHES_THRESHOLD = 10
    for i in range(len(images)):
        matches = getMatches(current_image_descriptors, descriptors[i])
        matches1, matches2 = matches
        max_matches1, max_matches2 = max_matches
        if len(matches1) > len(max_matches1) and len(matches1) > MATCHES_THRESHOLD:
            currH, numInliers = RANSAC(matches, current_image_keypoints, keypoints[i])
            column_average = np.average(current_image_keypoints[matches1][:, 1])
            if numInliers > maxInliers:
                max_matches = matches
                maxInliers = numInliers
                max_H = currH
                max_index = i
                if column_average < current_image.shape[1]/2:
                    isLeft = False
                else:
                    isLeft = True

    if max_index == -1:
        return np.empty(shape=(0,0)), -1

    if not isLeft:
        max_matches2, max_matches1 = max_matches
        max_matches = max_matches1, max_matches2
        max_H, _ = RANSAC(max_matches, keypoints[max_index], current_image_keypoints)
        return warpImageWithMapping(images[max_index], current_image, max_H), max_index
    return warpImageWithMapping(current_image, images[max_index], max_H), max_index

print(f"Length of Images: {len(images)}")
keypoints = []
descriptors = []
for i in range(len(images)):
    keypoints.append(detectKeypoints(images[i]))
    descriptors.append(computeDescriptors(images[i], keypoints[i]))

current_image = images[len(images)-1]
images.pop(len(images)-1)
keypoints.pop(len(keypoints)-1)
descriptors.pop(len(descriptors)-1)

foundStitch = False

num_images_to_stitch = len(images)
for i in range(num_images_to_stitch):
    print(f"Length of Images: {len(images)}")
    new_image, joined_index = find_and_stitch_closest_image(current_image, images, keypoints, descriptors)
    if new_image.shape[0] == 0 and not foundStitch:
        current_image = images[len(images) - 1]
        images.pop(len(images) - 1)
        keypoints.pop(len(keypoints) - 1)
        descriptors.pop(len(descriptors) - 1)
        continue
    if new_image.shape[0] == 0 and foundStitch:
        break
    print(f"Finished {i+1} stitch")
    current_image = new_image
    foundStitch = True
    images.pop(joined_index)
    keypoints.pop(joined_index)
    descriptors.pop(joined_index)

plt.imshow(current_image, cmap='gray', vmin=0, vmax=255)

plt.show()

