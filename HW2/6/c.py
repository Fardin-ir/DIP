import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from b import extract_foreground


def extract_foreground_mask(images, numFrames, test_fram, method=np.average, plot=False):
    foregrounds = extract_foreground(images, numFrames, test_fram, plot=False,method=method)
    foreground_masks = []
    trhs = [70,60,40,40]
    for i in range(len(foregrounds)):
        _, trh = cv2.threshold(foregrounds[i].astype('float32'), trhs[i], 255, cv2.THRESH_BINARY)
        foreground_masks.append(trh)
    if plot:
        plt.figure()
        for i in range(len(foreground_masks)):
            plt.subplot(2, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imsave(f"c/n{numFrames[i]}_method={method.__name__}.jpg", foreground_masks[i], cmap="gray")
            plt.imshow(foreground_masks[i], cmap="gray")
            plt.xlabel(f'numFrame = {numFrames[i]}, trh = {trhs[i]}, method={method.__name__}')
        plt.show()
    return foreground_masks

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
    extract_foreground_mask(images,[2,5,10,20],test_frame,plot=True,method=np.average)
    # call function np.median
    extract_foreground_mask(images, [2, 5, 10, 20], test_frame, plot=True, method=np.median)