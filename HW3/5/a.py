import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab as pl
from pypher.pypher import psf2otf


def img_dfft(img):
    if len(img.shape) == 2:
        img_dft = np.fft.fft2(img)
        img_dft = np.fft.fftshift(img_dft)
    else:
        channels = []
        for c in range(3):
            img_dft = np.fft.fft2(img[:,:,c])
            img_dft = np.fft.fftshift(img_dft)
            channels.append(img_dft)
        img_dft = np.dstack([channels[0],
                            channels[1],
                            channels[2]])
    return img_dft

def img_idft(f_shift):
    if len(f_shift.shape) == 2:
        f = np.fft.fftshift(f_shift)
        inv_img = np.fft.ifft2(f)
        inv_img = np.abs(inv_img)
        inv_img -= np.min(inv_img,axis=(0,1))
        inv_img = inv_img / np.max(inv_img,axis=(0,1)) * 255
        inv_img = inv_img.astype(np.uint8)
    else:
        channels = []
        for c in range(3):
            f = np.fft.fftshift(f_shift[:,:,c])
            inv_img = np.fft.ifft2(f)
            inv_img = np.abs(inv_img)
            inv_img -= np.min(inv_img, axis=(0, 1))
            inv_img = inv_img / np.max(inv_img, axis=(0, 1)) * 255
            inv_img = inv_img.astype(np.uint8)
            channels.append(inv_img)
        inv_img = np.dstack([channels[0].astype(int),
                            channels[1].astype(int),
                            channels[2].astype(int)])
    return inv_img

def import_imgs():
    donald1 = cv2.imread("P5/donald_in_car_1.png")
    donald2 = cv2.imread("P5/donald_in_car_2.png")
    donald1 = cv2.cvtColor(donald1, cv2.COLOR_BGR2GRAY)
    donald2 = cv2.cvtColor(donald2, cv2.COLOR_BGR2RGB)
    kernels = []
    for filename in glob.glob("P5/*.png"):
        if not filename.split("\\")[1].startswith("kernel"):
            continue
        kernel = cv2.imread(filename)
        kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
        kernels.append(kernel)
    return donald1,donald2,kernels

def applay_kernel(img, kernels):
    kernels_oft = [psf2otf(kernel,(img.shape[0],img.shape[1])) for kernel in kernels]
    if len(img.shape) == 3:
        kernels_oft = [np.stack((x,)*3, axis=-1) for x in kernels_oft]
    img_dft = img_dfft(img)
    blured_imgs = []
    for kernel in kernels_oft:
        blured_img_dft = img_dft * kernel
        blured_img = img_idft(blured_img_dft)
        blured_imgs.append(blured_img)
    return blured_imgs


if __name__ == "__main__":
    donald1, donald2, kernels = import_imgs()
    blured_trump1 = applay_kernel(donald1,kernels)
    for i in range(len(blured_trump1)):
        fig, axs = plt.subplots(1,2,gridspec_kw={'width_ratios': [3, 1]})
        axs[0].imshow(blured_trump1[i], cmap="gray")
        axs[1].imshow(kernels[i], cmap="gray")
        fig.tight_layout()
        plt.show()

    blured_trump2 = applay_kernel(donald2, kernels)
    for i in range(len(blured_trump2)):
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
        axs[0].imshow(blured_trump2[i])
        axs[1].imshow(kernels[i], cmap="gray")
        fig.tight_layout()
        plt.show()
