import glob
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab as pl
from a import import_imgs,img_dfft,img_idft,applay_kernel
from pypher.pypher import psf2otf
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def recover_image(img,h):
    H = psf2otf(h, (img.shape[0],img.shape[1]))
    if len(img.shape) == 3:
        H = [np.stack((x,)*3, axis=-1) for x in H]
    H = np.array(H)
    y = img_dfft(img)
    x = (H.conj() * y) / (H.conj() * H)
    x = img_idft(x)
    return x

if __name__ == "__main__":
    donald1, donald2, kernels = import_imgs()
    blured_trump1 = applay_kernel(donald1,kernels)
    blured_trump2 = applay_kernel(donald2, kernels)


    for i in range(len(blured_trump1)):
        plt.subplot(2,2,i+1)
        x = recover_image(blured_trump1[i], kernels[i])
        plt.imshow(x, cmap="gray")
        plt.xlabel(f"psnr:  {psnr(x, donald1):.2f}")
    plt.tight_layout()
    plt.show()

    for i in range(len(blured_trump2)):
        plt.subplot(2, 2, i + 1)
        x = recover_image(blured_trump2[i], kernels[i])
        plt.imshow(x, cmap="gray")
        plt.xlabel(f"psnr:  {psnr(x, donald2):.2f}")
    plt.tight_layout()
    plt.show()