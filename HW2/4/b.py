import numpy as np
import cv2
import matplotlib.pyplot as plt
from a import bitplane_slice


def xor(img1, img2, plot=False):
    # get bit slices of img1 and img2 with 'bitplane_slice' function
    bit_slices_img1 = bitplane_slice(img1)
    bit_slices_img2 = bitplane_slice(img2)
    bit_slices_xor = np.logical_xor(bit_slices_img1, bit_slices_img2)
    #plot xor
    if plot:
        plt.figure()
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(bit_slices_xor[i], cmap='gray')
            plt.xlabel(f'k = {i}')
        plt.show()
    return bit_slices_xor


if __name__ == '__main__':
    img_hw1 = cv2.imread('P4/highway_1.png')
    img_hw2 = cv2.imread('P4/highway_2.png')
    xor(img_hw1, img_hw2, True)
    img_p1 = cv2.imread('P4/pavement_1.png')
    img_p2 = cv2.imread('P4/pavement_2.png')
    xor(img_p1, img_p2, True)
