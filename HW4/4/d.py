import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage, misc

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

def wgif(p, I, r, r2, e, same_shape=True):
    if not(same_shape):
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)

    # size of kernel
    r = 2 * r - 1
    r2 = 2 * r2 - 1
    # cv2.blur is a mean filter
    mean_I = cv2.blur(I,(r,r))
    mean_I2 = cv2.blur(I,(r2,r2))
    mean_p = cv2.blur(p,(r,r))
    corr_I = cv2.blur(np.multiply(I,I),(r,r))
    corr_I2 = cv2.blur(np.multiply(I,I),(r2,r2))
    corr_p = cv2.blur(np.multiply(I,p),(r,r))
    var_I = corr_I - np.multiply(mean_I,mean_I)
    var_I2 = corr_I2 - np.multiply(mean_I2, mean_I2)
    cov_Ip = corr_p - np.multiply(mean_I,mean_p)
    landa = (0.001*1)**2
    si = (1 / (p.shape[0] * p.shape[1]) ) * (var_I2 + landa) * np.sum(np.reciprocal(var_I2+ landa))
    a = np.divide(cov_Ip,(var_I+e/si))
    b = mean_p - np.multiply(a, mean_I)
    mean_a = cv2.blur(a,(r,r))
    mean_b = cv2.blur(b,(r,r))
    q = np.multiply(mean_a,I) + mean_b
    # In case that same_shape == False, we might get out of bounds a little
    return q


if __name__ == "__main__":
    img = cv2.imread("P4/post-mortem_4.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask[mask>254] = 1
    mask[mask!=1] = 0
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    img = cv2.inpaint(img, mask.astype("uint8"), 20, cv2.INPAINT_TELEA)

    plt.imshow(img)
    plt.show()

    f_shift = np.zeros(img.shape,dtype="complex")
    for c in range(3):
        f_shift[:,:,c], f_img = img_dft(img[:,:,c])


    h = gaussian_kernel(f_shift.shape[0], f_shift.shape[1], 40, 400)
    h = (h - h.min()) / (h.max() - h.min())
    plt.imshow(h, cmap="gray")
    plt.show()

    i_image = np.zeros(img.shape)
    for c in range(3):
        f_filtered = h * f_shift[:,:,c]
        i_image[:,:,c] = img_idft(f_filtered,plot=False)

    plt.imshow(i_image/255)
    plt.show()