import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.measure import compare_ssim
from a import *

if __name__ == '__main__':
    date = cv2.imread('P8/date.png')
    # date/255 because in matplotlib, array of float should be in range [0,1]
    # gray version of image as guidance
    gray_date = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY) / 255
    date = cv2.cvtColor(date, cv2.COLOR_BGR2RGB) / 255
    # plot image
    plt.imshow(date)
    plt.show()
    # define list of parameters
    r_list = [2, 4, 6, 8, 10, 16]
    e_list = [0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2]


    guided_date = guided_filter(date, gray_date, r_list[1], e_list[0], False)
    plt.figure()
    plt.imshow(guided_date)
    plt.title(f'r={r_list[1]},e={e_list[0]:.5f}')
    plt.show()
    plt.imsave(f'e/g_r={r_list[1]},e={e_list[0]:.5f}.jpg', guided_date)
