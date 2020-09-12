import matplotlib.pyplot as plt
import cv2


def drawBlobs(im, blobs, numBlobsToDraw, title):
    
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #fig = plt.figure()
    fig,ax = plt.subplots(1)
    plt.title(title)

    order = (-blobs[:,-1]).argsort()

    ax.imshow(im, cmap='gray', vmin=0, vmax=255)

    for i in range(min(numBlobsToDraw, len(blobs))):
        blob = blobs[order[i]]
        x,y,r = blob[:3]
        ax.add_patch(plt.Circle((y, x), r, color='red', fill=False))
