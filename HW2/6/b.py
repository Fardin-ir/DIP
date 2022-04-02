import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from a import extract_background

def extract_foreground(images, numFrames, test_fram, method=np.average, plot=False):
    # extract backgrounds (part a)
    backgrounds = extract_background(images, numFrames, method=method)
    foregrounds = []
    for i in range(len(backgrounds)):
        sub = cv2.absdiff(test_fram, backgrounds[i])
        sub = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
        foregrounds.append(sub)
    if plot:
        plt.figure()
        for i in range(len(foregrounds)):
            plt.subplot(2, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imsave(f"b/n{numFrames[i]}_method={method.__name__}.jpg",foregrounds[i], cmap="gray")
            plt.imshow(foregrounds[i], cmap="gray")
            plt.xlabel(f'numFrame = {numFrames[i]}, method={method.__name__}')
        plt.show()
    return foregrounds

if __name__ == "__main__":
    #import all frames to images array
    filenames = glob.glob("P6/frames/*.png")
    filenames.sort()
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)
    # import test frame
    test_frame = cv2.imread('P6/test/football_test.png')
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    # call function np.average
    extract_foreground(images,[2,5,10,20],test_frame,plot=True,method=np.average)
    # call function np.median
    extract_foreground(images, [2, 5, 10, 20], test_frame, plot=True, method=np.median)

