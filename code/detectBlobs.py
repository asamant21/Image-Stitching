import numpy as np
import math
import cv2
from scipy import ndimage

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
#   num_intervals  - number of sigma values for each octave
#   threshold      - threshold for blob detection
#
# Ouput:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def generateLOGFilter(sigma: float):
    filterSize = 2*math.floor(3*sigma)+1
    logFilter = np.empty(shape=(filterSize, filterSize))
    logFilter.fill(0)
    center = (filterSize-1)/2
    for row in range(filterSize):
        for col in range(filterSize):
            xCoord = float(col - center)
            yCoord = float(row - center)
            logFilter[row, col] = laplacianOfGaussian(xCoord, yCoord, sigma)

    # Zero Out Sum of Laplacian
    logFilter -= logFilter.mean()

    return logFilter


def laplacianOfGaussian(x: float,y: float, sigma: float) -> float:
    sumOfCoord = x**2 + y**2
    sigmaTerm = 2*(sigma**2)
    expTerm = np.exp(-sumOfCoord/sigmaTerm)
    return (sumOfCoord-sigmaTerm)*expTerm/(sigma ** 4)


def findMaxLaplacianBlobs(octave, filter, img, threshold):
    resizeFactor = float(math.pow(2, octave))
    scaledImg = cv2.resize(img, (math.ceil(img.shape[1]/resizeFactor), math.ceil(img.shape[0]/resizeFactor)))
    convolvedImg = convolveImage(scaledImg, filter)
    #print(f"Max blob response before scaling: {np.max(convolvedImg)}")
    #print(np.average(convolvedImg))

    convolvedImg = cv2.resize(convolvedImg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    convolvedImg[convolvedImg <= threshold] = 0

    #print(f"Max blob response after scaling: {np.max(convolvedImg)}")
    effectiveFilterWidth = filter.shape[0]/2*math.pow(2, octave)
    convolvedImg = nonMaxSupression(convolvedImg, effectiveFilterWidth)
    return convolvedImg


def nonMaxSupression(img, filterSize):
    maxedImg = ndimage.maximum_filter(img, size=(filterSize, filterSize))
    diffImg = maxedImg - img
    img[diffImg != 0] = 0
    return img


def convolveImage(img, filter):
    convolvedImg = ndimage.convolve(img, filter, mode='nearest')
    return np.absolute(convolvedImg)

def DetectBlobs(
    im,
    sigma = 2,
    num_intervals = 12,
    threshold = 1e-4
    ):
    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # Determine number of octaves dynamically,
    # so that we can scale to find blobs of any size
    num_octaves = math.floor(math.log(min(im.shape)/sigma))

    sigmaMax = np.empty(shape=im.shape)
    normMax = np.empty(shape=im.shape)
    sigmaMax.fill(0)
    normMax.fill(0)
    for octave in range(num_octaves):  # octaves defines number times we want to scale
        for scale in range(num_intervals):
            logFilter = generateLOGFilter(sigma*math.pow(2, scale/num_intervals))
            convolvedResponse = findMaxLaplacianBlobs(octave, logFilter, im, threshold)
            maxDiff = convolvedResponse - normMax
            supressedLocs = maxDiff > 0
            normMax[supressedLocs] = convolvedResponse[supressedLocs]
            sigmaMax[supressedLocs] = sigma * math.pow(2, octave + scale / num_intervals)

    # Final NMS based on total size of Image
    finalSpacialWidth = 2*math.ceil(math.sqrt((0.006 * im.shape[0]) ** 2 + (0.006 * im.shape[1]) ** 2))
    maxedImg = ndimage.maximum_filter(normMax, size=(finalSpacialWidth, finalSpacialWidth))
    diffImg = maxedImg - normMax
    normMax[diffImg != 0] = 0
    sigmaMax[diffImg != 0] = 0
    blobs = np.array([])

    for row in range(normMax.shape[0]):
        for col in range(normMax.shape[1]):
            if normMax[row, col] > threshold:
                blobs = np.append(blobs, np.array([row, col, math.sqrt(2) * sigmaMax[row, col],
                                                   normMax[row, col]]))
    blobs = np.reshape(blobs, (-1, 4))
    return blobs.round()

