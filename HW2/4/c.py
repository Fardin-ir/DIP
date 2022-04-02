import numpy as np
import cv2
import matplotlib.pyplot as plt
from b import xor

def moving_regions(img1, img2, plot = False):
    bit_slices_xor = xor(img1, img2)
    #obtain gray scale image of moving regions
    result = np.sum(np.array([bit_slices_xor[k] * 2 ** k for k in range(4,8)]),axis=0)

    if plot:
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.show()
    return result

if __name__ == '__main__':
    img_hw1 = cv2.imread('P4/highway_1.png')
    img_hw2 = cv2.imread('P4/highway_2.png')
    moving_regions(img_hw1, img_hw2, True)
    img_p1 = cv2.imread('P4/pavement_1.png')
    img_p2 = cv2.imread('P4/pavement_2.png')
    moving_regions(img_p1, img_p2, True)
