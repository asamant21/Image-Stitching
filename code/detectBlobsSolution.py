# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.8.3 (default, May 19 2020, 13:54:14) 
# [Clang 10.0.0 ]
# Embedded file name: /Users/mingzhe/Downloads/COS429_Fall2020_HW1 (2)/code/detectBlobs.py
# Compiled at: 2020-09-05 21:36:47
# Size of source mod 2**32: 5476 bytes
import sys, numpy as np, math, cv2
from scipy import ndimage

def get_log_kernel(siz, std):
    x = y = np.linspace(-siz, siz, 2 * siz + 1)
    x, y = np.meshgrid(x, y)
    arg = -(x ** 2 + y ** 2) / (2 * std ** 2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h / h.sum() if h.sum() != 0 else h
    h1 = h * (x ** 2 + y ** 2 - 2 * std ** 2) / std ** 4
    return h1 - h1.mean()


def imresize(im, scale):
    h, w = im.shape
    new_size = (math.ceil(w * scale), math.ceil(h * scale))
    return cv2.resize(im, new_size)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def scaleSpace(im, num_intervals, minSize):
    pyra = {}
    pyra['size'] = im.shape
    im = imresize(im, 2)
    smallDim = min(im.shape)
    numOctaves = math.ceil(math.log(smallDim / minSize)) + 1
    pyra['im'] = []
    pyra['scale'] = np.zeros(numOctaves * num_intervals)
    stepSize = 2 ** (1 / num_intervals)
    currScale = 2
    offset = 0
    for s in range(numOctaves):
        for i in range(num_intervals):
            pyra['im'].append(imresize(im, 1 / stepSize ** i))
            pyra['scale'][offset + i] = currScale / stepSize ** i

        im = imresize(im, 0.5)
        currScale = currScale * 0.5
        offset = offset + num_intervals

    return pyra


def DetectBlobs(im, sigma=2, num_intervals=12, threshold=0.0001):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    r = math.ceil(3 * sigma)
    g = get_log_kernel(r, sigma)
    pyra = scaleSpace(im, num_intervals, g.shape[0])
    scores = np.zeros(shape=(im.shape[0], im.shape[1], len(pyra['scale'])))
    for i in range(len(pyra['scale'])):
        score0 = abs(ndimage.convolve((pyra['im'][i]), g, mode='nearest'))
        l = g.shape[0] // 2
        score = score0[l:-l, l:-l]
        score = np.pad(score, r, pad_with, padder=0)
        scores[:, :, i] = cv2.resize(score, (im.shape[1], im.shape[0]))

    nbhd = (3, 3)
    for i in range(len(pyra['scale'])):
        ordscore = ndimage.rank_filter((scores[:, :, i]), rank=(-1), size=nbhd)
        ismax = abs(ordscore - scores[:, :, i]) < sys.float_info.epsilon
        scores[:, :, i] = ismax * scores[:, :, i]

    blobScale = scores.argmax(axis=2)
    blobScore = np.take_along_axis(scores, (blobScale[:, :, None]), axis=2).reshape(im.shape)
    nmsRadius = max(1, math.ceil(0.005 * math.sqrt(im.shape[0] ** 2 + im.shape[1] ** 2)))
    nbhd = (2 * nmsRadius + 1, 2 * nmsRadius + 1)
    ordscore = ndimage.rank_filter(blobScore, rank=(-1), size=nbhd)
    ismax = abs(ordscore - blobScore) < sys.float_info.epsilon
    blobScore = blobScore * ismax
    inds = blobScore > threshold
    indices = np.argwhere(inds)
    r = 1 / pyra['scale'][blobScale[inds]].T * sigma * math.sqrt(2)
    blobs = np.concatenate((indices, r.reshape(-1, 1), blobScore[inds].reshape(-1, 1)), axis=1)
    return blobs