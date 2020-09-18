import os, sys
import math
import cv2
import random
import numpy as np
from detectBlobs import DetectBlobs

def detectKeypoints(im):
    im = im/255.0
    return DetectBlobs(im)


def computeDescriptors(im, keypoints):
    return computeSIFTDescriptors(im, keypoints)


def computeSIFTDescriptors(im, keypoints):
    kp = []
    for blob in keypoints:
        kp.append(cv2.KeyPoint(blob[1], blob[0], _size=blob[2]*2, _response=blob[-1], _class_id=len(kp)))
    detector = cv2.xfeatures2d_SIFT.create()
    return detector.compute(im, kp)[1]


def getMatches(descriptors1, descriptors2):
    descriptors1_matches = []
    descriptors2_matches = []
    THRESHOLD = 100

    distances = np.linalg.norm(descriptors1[:, None] - descriptors2, axis=2)
    for i in range(len(distances)):
        near_dist = np.argpartition(distances[i], 2)
        closest, second_closest = distances[i][near_dist[: 2]]
        if closest < second_closest * 0.75 and closest < THRESHOLD:
            descriptors1_matches.append(i)
            descriptors2_matches.append(near_dist[: 1][0])
    return np.array(descriptors1_matches), np.array(descriptors2_matches)


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
        rand_matches = random.sample(range(len(matches1)), S)
        H = fitH(keypoints1, keypoints2, matches, rand_matches)
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
        #print(f"Num inliers: {len(inliers)}")
        if len(inliers) > max_inliers:
            max_H = fitH(keypoints1, keypoints2, matches, inliers)
            max_inliers = len(inliers)
            E = min(E, 1 - len(inliers) / len(matches1))
            if E == 0:
                break
            N = recomputeN(P, E, S)
            #print(f"Max num_iterations: {N}")
        num_of_trials+=1

    return max_H, max_inliers



def warpImageWithMapping(im_left, im_right, H):
    top_left = H.dot(np.array([0, 0, 1]))
    top_left = np.divide(top_left, top_left[2])

    top_right = H.dot(np.array([im_left.shape[1] - 1, 0, 1]))
    top_right = np.divide(top_right, top_right[2])

    bottom_left = H.dot(np.array([0, im_left.shape[0] - 1, 1]))
    bottom_left = np.divide(bottom_left, bottom_left[2])

    bottom_right = H.dot(np.array([im_left.shape[1] - 1, im_left.shape[0] - 1, 1]))
    bottom_right = np.divide(bottom_right, bottom_right[2])

    min_col = abs(min(int(top_left[0]), int(top_right[0]), int(bottom_left[0]), int(bottom_right[0])))
    min_row = abs(min(int(top_left[1]), int(top_right[1]), int(bottom_left[1]), int(bottom_right[1])))

    max_col = abs(max(int(top_left[0]), int(top_right[0]), int(bottom_left[0]), int(bottom_right[0])))
    max_row = abs(max(int(top_left[1]), int(top_right[1]), int(bottom_left[1]), int(bottom_right[1])))

    final_col_len = min_col + max(max_col, im_right.shape[1])
    final_row_len = min_row + max(max_row, im_right.shape[0])
    translation = np.array([[1, 0, min_col], [0, 1, min_row], [0, 0, 1]])
    H = np.matmul(translation, H)

    stitch_image = cv2.warpPerspective(im_left, dsize=(final_col_len, final_row_len), M=H)

    stitch_image[min_row:min_row + im_right.shape[0], min_col:min_col + im_right.shape[1]] = im_right
    return stitch_image


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
    cv2.imshow(title, im_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
