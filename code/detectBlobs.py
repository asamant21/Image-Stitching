import numpy as np
import math
import cv2
from scipy import ndimage

# This code is taken and converted to Python from:
#
#   CMPSCI 670: Computer Vision, Fall 2014
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
# Part1:
#
#   DetectBlobs(...) detects blobs in the image using the Laplacian
#   of Gaussian filter. Blobs of different size are detected by scaling sigma
#   as well as the size of the filter or the size of the image. Downsampling
#   the image will be faster than upsampling the filter, but the decision of
#   how to implement this function is up to you.
#
#   For each filter scale or image scale and sigma, you will need to keep track of
#   the location and matching score for every blob detection. To combine the 2D maps
#   of blob detections for each scale and for each sigma into a single 2D map of
#   blob detections with varying radii and matching scores, you will need to use
#   Non-Max Suppression (NMS).
#
#   Additional Notes:
#       - We greyscale the input image for simplicity
#       - For a simple implementation of Non-Max-Suppression, you can suppress
#           all but the most likely detection within a sliding window over the
#           2D maps of blob detections (ndimage.maximum_filter may help).
#           To combine blob detections into a single 2D output,
#           you can take the max along the sigma and scale axes. If there are
#           still too many blobs detected, you can do a final NMS. Remember to
#           keep track of the blob radii.
#       - A tip that may improve your LoG filter: Normalize your LoG filter
#           values so that your blobs detections aren't biased towards larger
#           filters sizes
#
#   You can qualitatively evaluate your code using the evalBlobs.py script.
#
# Input:
#   im             - input image
#   sigma          - base sigma of the LoG filter
#   num_intervals  - number of sigma values for each filter size
#   threshold      - threshold for blob detection
#
# Ouput:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def DetectBlobs(
    im,
    sigma = 2,
    num_intervals = 12,
    threshold = 1e-4
    ):

    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # YOUR CODE STARTS HERE

    blobs = np.array([[im.shape[0]*0.5, im.shape[1]*0.5, 0.25*min(im.shape[0], im.shape[1]), 1]])

    return blobs.round()
