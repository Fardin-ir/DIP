import cv2
import matplotlib.pyplot as plt
import numpy as np
from align_imgs import align_imgs

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


if __name__ == "__main__":
    trump1 = cv2.imread("P3/Images/joe_1.png")
    trump2 = cv2.imread("P3/Images/joe_4.png")
    trump1 = cv2.cvtColor(trump1, cv2.COLOR_BGR2GRAY)
    trump2 = cv2.cvtColor(trump2, cv2.COLOR_BGR2GRAY)

    trump1, trump2 = align_imgs(trump1, trump2)
    trump1 = np.array(trump1)
    trump2 = np.array(trump2)
    plt.imshow(trump1, cmap="gray")
    plt.show()
    plt.imshow(trump2, cmap="gray")
    plt.show()
    img_dft(trump1,True)
    img_dft(trump2,True)