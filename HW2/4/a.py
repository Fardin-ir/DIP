import numpy as np
import cv2
import matplotlib.pyplot as plt

def bitplane_slice(img, plot = False):
    # convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bit_slices[k] contain bit value of bit number k
    bit_slices = []
    for k in range(8):
        bit_slices.append((img_gray // 2 ** k) % 2)
    # plot each bit slices
    if plot:
        plt.figure()
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(bit_slices[i],cmap='gray')
            plt.xlabel(f'k = {i}')
        plt.show()
    return np.array(bit_slices)

if __name__ == '__main__':
    img = cv2.imread('P4/highway_1.png')
    bitplane_slice(img, True)
    img = cv2.imread('P4/highway_2.png')
    bitplane_slice(img, True)
    img = cv2.imread('P4/pavement_1.png')
    bitplane_slice(img, True)
    img = cv2.imread('P4/pavement_2.png')
    bitplane_slice(img, True)