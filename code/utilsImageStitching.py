import os, sys
import math
import cv2
import random
import numpy as np
from sklearn.neighbors import KDTree
from detectBlobs import DetectBlobs

random.seed(42)
# detectKeypoints(...): Detect feature keypoints in the input image
#   You can either reuse your blob detector from part 1 of this assignment
#   or you can use the provided compiled blob detector detectBlobsSolution.pyc
#
#   Input: 
#        im  - input image
#   Output: 
#        detected feature points (in any format you like).


def detectKeypoints(im):
    im = im/255.0
    return DetectBlobs(im)




# computeDescriptors(...): compute descriptors from the detected keypoints
#   You can build the descriptors by flatting the pixels in the local 
#   neighborhood of each keypoint, or by using the SIFT feature descriptors from
#   OpenCV (see computeSIFTDescriptors(...)). Use the detected blob radii to
#   define the neighborhood scales.
#
#   Input:
#        im          - input image
#        keypoints   - detected feature points
#
#   Output:
#        descriptors - n x dim array, where n is the number of keypoints 
#                      and dim is the dimension of each descriptor. 
#
def computeDescriptors(im, keypoints):
    return computeSIFTDescriptors(im, keypoints)


# computeSIFTDescriptors(...): compute SIFT feature descriptors from the
#   detected keypoints. This function is provided to you.
#
#   Input:
#        im          - H x W array, the input image
#        keypoints   - n x 4 array, where there are n blobs detected and
#                      each row is [x, y, radius, score]
#
#   Output:
#        descriptors - n x 128 array, where n is the number of keypoints
#                      and 128 is the dimension of each descriptor.
#
def computeSIFTDescriptors(im, keypoints):
    kp = []
    for blob in keypoints:
        kp.append(cv2.KeyPoint(blob[1], blob[0], _size=blob[2]*2, _response=blob[-1], _class_id=len(kp)))
    detector = cv2.xfeatures2d_SIFT.create()
    return detector.compute(im, kp)[1]



# getMatches(...): match two groups of descriptors.
#
#   There are several strategies you can use to match keypoints from the left
#   image to those in the right image. Feel free to use any (or combinations
#   of) strategies:
#
#   - Return all putative matches. You can select all pairs whose
#   descriptor distances are below a specified threshold,
#   or select the top few hundred descriptor pairs with the
#   smallest pairwise distances.
#
#   - KNN-Match. For each keypoint in the left image, you can simply return the
#   the K best pairings with keypoints in the right image.
#
#   - Lowe's Ratio Test. For each pair of keypoints to be returned, if the
#   next best match with the same left keypoint is nearly as good as the
#   current match to be returned, then this match should be thrown out.
#   For example, given point A in the left image and its best and second best
#   matches B and C in the right image, we check: score(A,B) < score(A,C)*0.75
#   If this test fails, don't return pair (A,B)
#
#
#   Input:
#         descriptors1 - the descriptors of the first image
#         descriptors2 - the descriptors of the second image
# 
#   Output: 
#         index1       - 1-D array contains the indices of descriptors1 in matches
#         index2       - 1-D array contains the indices of descriptors2 in matches

def getMatches(descriptors1, descriptors2):

    descriptors1_matches = []
    descriptors2_matches = []
    THRESHOLD = 100

    kdt = KDTree(descriptors2, metric='euclidean')
    dists, matches = kdt.query(descriptors1, k=2)

    for i in range(len(matches)):
        if dists[i][0] < dists[i][1] * 0.75 and dists[i][0] < THRESHOLD:
            descriptors1_matches.append(i)
            descriptors2_matches.append(matches[i][0])
    print(f"Found {len(descriptors1_matches)} matches")
    return np.array(descriptors1_matches), np.array(descriptors2_matches)


"""
for i in range(len(descriptors1)):
    largest = -1
    second_largest = -1

    largest_index = -1
    second_largest_index = -1

    for j in range(len(descriptors2)):
        k = np.linalg.norm(np.array(descriptors1[i]) - np.array(descriptors2[j]))
        if k < largest or largest == -1:
                second_largest = largest
                second_largest_index = largest_index
                largest = k
                largest_index = j
        elif k < second_largest or second_largest == -1:
                second_largest = k
                second_largest_index = j
    if largest < second_largest * 0.75 and largest < THRESHOLD:
        descriptors1_matches.append(i)
        descriptors2_matches.append(largest_index)
"""



# RANSAC(...): run the RANSAC algorithm to estimate a homography mapping between two images.
#   Input:
#        matches - two 1-D arrays that contain the indices on matches. 
#        keypoints1       - keypoints on the left image
#        keypoints2       - keypoints on the right image
#
#   Output:
#        H                - 3 x 3 array, a homography mapping between two images
#        numInliers       - int, the number of inliers 
#
#   Note: Use four matches to initialize the homography in each iteration.
#         You should output a single transformation that gets the most inliers 
#         in the course of all the iterations. For the various RANSAC parameters 
#         (number of iterations, inlier threshold), play around with a few 
#         "reasonable" values and pick the ones that work best.

def recomputeN(P, E, S):
    return math.log(1 - P) / math.log(1 - (1 - E)**S)


def fitH(keypoints1, keypoints2, matches, sample):
    matches1, matches2 = matches

    A = []
    for i in range(len(sample)):
        y_1 = keypoints1[matches1[sample[i]]][0]
        x_1 = keypoints1[matches1[sample[i]]][1]

        y_2 = keypoints2[matches2[sample[i]]][0]
        x_2 = keypoints2[matches2[sample[i]]][1]

        A.append([0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2])
        A.append([x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2])

    eVal, eVec = np.linalg.eig(np.matmul(np.transpose(A), A))
    return np.reshape(eVec[:, np.argmin(eVal)], (-1, 3))


def RANSAC(matches, keypoints1, keypoints2):
    S = 4
    N = sys.maxsize
    P = 0.99
    INLIER_THRESHOLD = 5 # should be in [1, 5]
    matches1, matches2 = matches

    num_of_trials = 0
    max_inliers = 0
    max_H = []
    E = 1
    while num_of_trials < N:
        print(N)
        rand_matches = random.sample(range(len(matches1)), S)
        H = fitH(keypoints1, keypoints2, matches, rand_matches)
        print("Done fitting H")
        inliers = []
        for i in range(len(matches1)):
            y_1 = keypoints1[matches1[i]][0]
            x_1 = keypoints1[matches1[i]][1]

            y_2 = keypoints2[matches2[i]][0]
            x_2 = keypoints2[matches2[i]][1]

            H_p = H.dot(np.array([x_1, y_1, 1]))
            H_p = np.divide(H_p, H_p[2])
            if np.linalg.norm(H_p - np.array([x_2, y_2, 1])) < INLIER_THRESHOLD:
                inliers.append(i)
        print(f"Num inliers: {len(inliers)}")
        if len(inliers) > max_inliers:
            max_H = fitH(keypoints1, keypoints2, matches, inliers)
            max_inliers = len(inliers)
            E = min(E, 1 - len(inliers) / len(matches1))
            N = recomputeN(P, E, S)
            print(f"Max num_iterations: {N}")
        num_of_trials+=1

    return max_H, max_inliers




# warpImageWithMapping(...): warp one image using the homography mapping and
#   composite the warped image and another image into a panorama.
# 
#   Input: 
#        im_left, im_right - input images.
#        H                 - 3 x 3 array, a homography mapping
#  
#   Output:
#        Panorama made of the warped image and the other.
#
#       To display the full warped image, you may want to modify the matrix H.
#       CLUE: first get the coordinates of the corners under the transformation,
#             use the new corners to determine the offsets to move the
#             warped image such that it can be displayed completely.
#             Modify H to fulfill this translate operation.
#       You can use cv2.warpPerspective(...) to warp your image using H

def warpImageWithMapping(im_left, im_right, H):
    top_left = H.dot(np.array([0, 0, 1]))
    top_left = np.divide(top_left, top_left[2])

    top_right = H.dot(np.array([0, im_left.shape[1] - 1, 1]))
    top_right = np.divide(top_right, top_right[2])

    bottom_left = H.dot(np.array([im_left.shape[0] - 1, 0, 1]))
    bottom_left = np.divide(bottom_left, bottom_left[2])

    bottom_right = H.dot(np.array([im_left.shape[0] - 1, im_left.shape[1] - 1, 1]))
    bottom_right = np.divide(bottom_right, bottom_right[2])

    print(top_left)
    print(top_right)
    print(bottom_left)
    print(bottom_right)

    new_image = cv2.warpPerspective(im_left, dsize=(im_left.shape[1], im_left.shape[0]), M=H)
    #new_image = np.empty((max(im_left.shape[0], im_right.shape[0]), im_left.shape[1]+im_right.shape[1]), dtype=np.uint8)
    #new_image[:im_left.shape[0], :im_left.shape[1]] = im_left
    #new_image[:im_right.shape[0], im_left.shape[1]:] = im_right
    cv2.imshow("Warped", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# drawMatches(...): draw matches between the two images and display the image.
#
#   Input:
#         im1: input image on the left
#         im2: input image on the right
#         matches: (1-D array, 1-D array) that contains indices of descriptors in matches
#         keypoints1: keypoints on the left image
#         keypoints2: keypoints on the right image
#         title: title of the displayed image.
#
#   Note: This is a utility function that is provided to you. Feel free to
#   modify the code to adapt to the keypoints and matches in your own format.

def drawMatches(im1, im2, matches, keypoints1, keypoints2, title='matches'):
    idx1, idx2 = matches

    cv2matches = []
    for i,j in zip(idx1, idx2):
        cv2matches.append(cv2.DMatch(i, j, _distance=0))

    _kp1, _kp2 = [], []
    for i in range(len(keypoints1)):
        _kp1.append(cv2.KeyPoint(keypoints1[i][1], keypoints1[i][0], _size=keypoints1[i][2], _response=keypoints1[i][3], _class_id=len(_kp1)))
    for i in range(len(keypoints2)):
        _kp2.append(cv2.KeyPoint(keypoints2[i][1], keypoints2[i][0], _size=keypoints2[i][2], _response=keypoints2[i][3], _class_id=len(_kp2)))
    
    im_matches = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(im1, _kp1, im2, _kp2, cv2matches, im_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow(title, im_matches)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

