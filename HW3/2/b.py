import numpy as np
import matplotlib.pyplot as plt
import cv2
from a import img_dft

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
    img = cv2.imread("P2/donald_x-ray.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f_shift,f_img = img_dft(img)

    h = gaussian_kernel(f_shift.shape[0], f_shift.shape[1], 40, 400)
    #h = cv2.imread("Filter.jpg")
    #h = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
    h = (h - h.min()) / (h.max() - h.min())
    print(h.min())
    plt.imshow(h, cmap="gray")
    plt.show()

    f_filtered = h * f_shift
    img_idft(f_filtered)
