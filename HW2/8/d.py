import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim
from a import get_noisy_image


if __name__ == '__main__':
    aush = cv2.imread('P8/aush.png')
    # aush/255 because in matplotlib, array of float should be in range [0,1]
    aush = cv2.cvtColor(aush, cv2.COLOR_BGR2GRAY)/255
    # add noise to aush!
    noisy_aush = get_noisy_image(aush,0.1)
    # plot and save noisy image
    plt.figure()
    plt.imshow(noisy_aush, cmap='gray')
    plt.imsave('d/noisy_aush.jpg',noisy_aush, cmap='gray')
    plt.show()
    # define list of parameters
    size = 3
    sigma = 1
    # cv2.GaussianBlur is Gaussian smoothing filter
    filtered_aush = cv2.GaussianBlur(noisy_aush,(size,size),sigma)
    plt.figure()
    plt.imshow(filtered_aush, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    # show parameters as x-label
    plt.xlabel(f'size={size}, sigma={sigma}')
    # calculate and display ssim and psnr as images title
    psnr = cv2.PSNR(aush, filtered_aush)
    ssim = compare_ssim(aush, filtered_aush)
    plt.title(f'psnr={psnr:.2f}, ssim={ssim:.2f}')
    # save images with appropriate name
    plt.imsave(f'd/f_size={size},sigma={sigma}.jpg',filtered_aush, cmap='gray')
    plt.show()