# Evaluation code for blob detector
#
# Your goal is to implement the laplacian of the gaussian blob detector 
#
# This code is taken and converted to Python from:
#
#   CMPSCI 670: Computer Vision, Fall 2014
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 3: Blob detector

import os, sys
import cv2
import numpy as np

from detectBlobs import DetectBlobs
from drawBlobs import drawBlobs
import matplotlib.pyplot as plt

image_path = '../data/uttower_right.jpg'
numBlobsToDraw = 3000

im = cv2.imread(image_path)

detected_blobs = DetectBlobs(im)
print ('Detect %d blobs' % (detected_blobs.shape[0]))

drawBlobs(im, detected_blobs, numBlobsToDraw, 'DetectBlobs')

plt.show()
