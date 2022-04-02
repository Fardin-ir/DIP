import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_ssim
from a import *

if __name__ == '__main__':
    zoolbia = cv2.imread('P8/zoolbia_bamieh.png')
    # zoolbia/255 because in matplotlib, array of float should be in range [0,1]
    zoolbia = cv2.cvtColor(zoolbia, cv2.COLOR_BGR2RGB) / 255
    # get guided_filter of zoolbia(g) and plot is
    guided_zoolbia = guided_filter(zoolbia, zoolbia, 4, 0.2**2)
    plt.figure()
    plt.imshow(guided_zoolbia)
    plt.show()
    # save guided_filter of zoolbia
    r, e = 4, 0.04
    plt.imsave(f'h/g_r={r},e={e}.jpg', guided_zoolbia)
    # calculate d
    d = cv2.subtract(zoolbia, guided_zoolbia)
    # list for different value of a
    a_list = [3,5,10]
    for i in range(len(a_list)):
        # calculate enhanced image
        p_enh = cv2.add(guided_zoolbia , a_list[i] * d)
        p_enh = np.clip(p_enh,0,1)
        plt.figure()
        plt.imshow(p_enh)
        plt.xlabel(f'Enhanced Image ,r={r},e={e},a={a_list[i]}')
        plt.show()
        plt.imsave(f'h/g_r={r},e={e},a={a_list[i]}.jpg',p_enh)


