import glob
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pylab as pl
from a import import_imgs,img_dfft,img_idft,applay_kernel
from pypher.pypher import psf2otf
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import fftn, ifftn

def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size

    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(psf.imag)) / np.max(abs(psf)) <= num_small:
            psf = psf.real

            # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0] / 2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1] / 2)), axis=1)

        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def recover_image(img,blured_image,landa,size):
    A = np.eye(img.shape[0])
    A = cv2.resize(A, (img.shape[1], img.shape[0]))
    A = img_dfft(A)
    blured_image = img_dfft(blured_image)
    if len(img.shape) == 3:
        A = [np.stack((x,) * 3, axis=-1) for x in A]
        A = np.array(A)
    y = img_dfft(img)
    H = (blured_image.conj() * y) / (blured_image.conj() * blured_image + landa*(A.conj() * A))
    if len(img.shape) == 3:
        H = H[:,:,0]
    h = otf2psf(H,size)
    return h

if __name__ == "__main__":
    donald1, donald2, kernels = import_imgs()
    blured_trump1 = applay_kernel(donald1,kernels)
    blured_trump2 = applay_kernel(donald2, kernels)

    landa = 0.001
    for i in range(len(blured_trump1)):
        plt.subplot(2, 2, i + 1)
        x = recover_image(blured_trump1[i], donald1, landa, kernels[i].shape)
        norm_kernel = (kernels[i] - kernels[i].min())/(kernels[i].max() - kernels[i].min())
        plt.imshow(np.abs(x), cmap="gray")
        plt.xlabel(f"RMSE:  {rmse(np.abs(x), norm_kernel):.2f}")
    plt.tight_layout()
    plt.show()

    for i in range(len(blured_trump2)):
        plt.subplot(2, 2, i + 1)
        x = recover_image(blured_trump2[i], donald2, landa, kernels[i].shape)
        norm_kernel = (kernels[i] - kernels[i].min()) / (kernels[i].max() - kernels[i].min())
        plt.imshow(np.abs(x), cmap="gray")
        plt.xlabel(f"RMSE:  {rmse(np.abs(x), norm_kernel):.2f}")
    plt.tight_layout()
    plt.show()