import numpy as np
import cv2
import matplotlib.pyplot as plt
from c import moving_regions


def enhancement(img):
    # use median filtering to remove small details
    box = cv2.medianBlur(img.astype('float32'), 5)
    # use thresholding to enhance contrast
    _, trh = cv2.threshold(box.astype('float32'), 40, 255, cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(trh, cmap="gray")
    plt.show()


if __name__ == '__main__':
    img_hw1 = cv2.imread('P4/highway_1.png')
    img_hw2 = cv2.imread('P4/highway_2.png')
    mv = moving_regions(img_hw1, img_hw2, True)
    enhancement(mv)
    img_p1 = cv2.imread('P4/pavement_1.png')
    img_p2 = cv2.imread('P4/pavement_2.png')
    mv = moving_regions(img_p1, img_p2, True)
    enhancement(mv)
