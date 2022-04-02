import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_dft(img, plot=False):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_abs = np.abs(f_shift) + 1
    f_bounded = 20 * np.log10(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    if plot:
        plt.imshow(f_img, cmap="gray")
        plt.title("Image spectrum")
        plt.show()
    return f_shift,f_img

def img_idft(f_shift, plot=True):
    f = np.fft.fftshift(f_shift)
    inv_img = np.fft.ifft2(f)
    inv_img = np.abs(inv_img)
    inv_img -= np.min(inv_img)
    inv_img = inv_img / np.max(inv_img) * 255
    inv_img = inv_img.astype(np.uint8)
    if plot:
        plt.imshow(inv_img, cmap="gray")
        plt.title("Inverse image")
        plt.show()
    return inv_img

def gaussian_kernel(dimension_x, dimension_y, sigma_x, sigma_y):
    x = cv2.getGaussianKernel(dimension_x, sigma_x)
    y = cv2.getGaussianKernel(dimension_y, sigma_y)
    kernel = x.dot(y.T)
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    return kernel

if __name__ == "__main__":
    img = cv2.imread("P1/nasir_receiving_pachekhari.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    plt.show()
    f_shift, f_img = img_dft(img, plot=True)
    h = gaussian_kernel(f_shift.shape[0], f_shift.shape[1], 50, 400)
    h = (h - h.min()) / (h.max() - h.min())
    plt.imshow(h, cmap="gray")
    plt.show()

    f_filtered = h * f_shift
    img_idft(f_filtered)
