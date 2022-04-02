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
    else:
        channels = []
        for c in range(3):
            f = np.fft.fftshift(f_shift[:,:,c])
            inv_img = np.fft.ifft2(f)
            inv_img = np.abs(inv_img)
            channels.append(inv_img)
        inv_img = np.dstack([channels[0].astype(int),
                            channels[1].astype(int),
                            channels[2].astype(int)])
    return inv_img

def gaussian_kernel(dimension_x, dimension_y, sigma, d):
    x = cv2.getGaussianKernel(dimension_x, sigma)
    y = cv2.getGaussianKernel(dimension_y, sigma)
    kernel = x.dot(y.T)
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    if d == 3:
        kernel = np.dstack([kernel, kernel, kernel])
    return kernel

def gaussian_kernel1(dimension_x, dimension_y, sigma, d):
    gaussian_filter = np.zeros((dimension_x, dimension_y))
    a = dimension_x // 2
    b = dimension_y // 2
    for x in range(-a, a):
        for y in range(-b, b):
            x1 = np.sqrt(2 * np.pi * (sigma ** 2))
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + a - 1, y + b - 1] = x2
    return gaussian_filter

def normalize(a):
    return (a - np.min(a,axis=(0,1))) / (np.max(a,axis=(0,1)) - np.min(a,axis=(0,1)))

def get_laplacian_set(img, levels):
    laplacian_set = []
    for n in levels:
        dft = img_dfft(img)
        g_l = img_idft(dft * gaussian_kernel(img.shape[0], img.shape[1], 2 ** (n-1), len(img.shape)))
        g_l_1 = img_idft(dft * gaussian_kernel(img.shape[0], img.shape[1], 2 ** (n), len(img.shape)))
        l = g_l - g_l_1
        laplacian_set.append(l)
    return laplacian_set

def get_gaussian_set(img, levels):
    gaussian_set = []
    for n in levels:
        dft = img_dfft(img)
        g_l = img_idft(dft * gaussian_kernel(img.shape[0], img.shape[1], 2 ** (n-1), len(img.shape)))
        gaussian_set.append(g_l)
    return gaussian_set


if __name__ == "__main__":
    levels = range(1,9)
    img1 = cv2.imread("P6/I/audi_q7.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_laplacian_set = get_laplacian_set(img1, levels)
    img2 = cv2.imread("P6/I/saipa_151.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_laplacian_set = get_laplacian_set(img2, levels)
    mask = cv2.imread("P6/I/audi_saipa_mask.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)/255
    mask_gaussian_set = get_gaussian_set(mask, levels)

    C = []
    for i in range(len(levels)):
        c =  (mask_gaussian_set[i] * img1_laplacian_set[i] + (1 - mask_gaussian_set[i])*img2_laplacian_set[i])
        C.append(c)

    for i,x in enumerate([img1_laplacian_set,img2_laplacian_set,mask_gaussian_set,C]):
        for j,img in enumerate(x):
            plt.subplot(4, 8, i*8 + j+1)
            plt.xticks([])
            plt.yticks([])
            if i in [0,1,3]:
                img = 1 - normalize(img)
            plt.imshow(img, cmap="gray")
    plt.tight_layout(h_pad=0, w_pad=0, pad =0)
    plt.show()

    result = C[0]
    for i in range(1,len(levels)):
        result += C[i]

    result =1 - normalize(result)
    plt.imshow(result, cmap="gray")
    plt.title(f"n={levels[-1]}")
    plt.show()

