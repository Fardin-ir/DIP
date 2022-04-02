import cv2
import matplotlib.pyplot as plt
import numpy as np
from align_imgs import align_imgs
from a import img_dft

def img_idft(f_shift, plot=True):
    f = np.fft.fftshift(f_shift)
    inv_img = np.fft.ifft2(f)
    inv_img = np.abs(inv_img)
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

def hybrid_images(img1,img2,d1, d2):
    img1_dft, _ = img_dft(img1, False)
    img2_dft, _ = img_dft(img2, False)
    filter = gaussian_kernel(img1_dft.shape[0], img1_dft.shape[1], d1, d1)
    filtered_img1 = img_idft(img1_dft * filter, False)
    filtered_img1 = (filtered_img1 - filtered_img1.min()) / (filtered_img1.max() - filtered_img1.min()) * 255
    filter = gaussian_kernel(img1_dft.shape[0], img1_dft.shape[1], d2, d2)
    filtered_img2 = np.abs(img2 - img_idft(img2_dft * filter, False))
    filtered_img2 = (filtered_img2 - filtered_img2.min()) / (filtered_img2.max() - filtered_img2.min()) * 255
    return filtered_img1, filtered_img2

if __name__ == "__main__":
    trump1 = cv2.imread("P3/Images/joe_1.png")
    trump2 = cv2.imread("P3/Images/joe_4.png")
    trump1 = cv2.cvtColor(trump1, cv2.COLOR_BGR2GRAY)
    trump2 = cv2.cvtColor(trump2, cv2.COLOR_BGR2GRAY)
    trump1,trump2 = align_imgs(trump1,trump2)
    trump2 = cv2.resize(trump2,(trump1.shape[1],trump1.shape[0]))

    filtered_trump1, filtered_trump2 = hybrid_images(trump1,trump2,30,30)

    plt.figure()
    plt.imshow(filtered_trump1, cmap="gray")
    plt.title("First Image - Low Pass Filter")
    plt.show()
    plt.figure()
    plt.imshow(filtered_trump2, cmap="gray")
    plt.title("Second Image - High Pass Filter")
    plt.show()

    _,f_image1 = img_dft(filtered_trump1,False)
    plt.figure()
    plt.imshow(f_image1, cmap="gray")
    plt.title("First Image spectrum- Low Pass Filter")
    plt.show()
    _,f_image2 = img_dft(filtered_trump2,False)
    plt.figure()
    plt.imshow(f_image2, cmap="gray")
    plt.title("Second Image spectrum- High Pass Filter")
    plt.show()
