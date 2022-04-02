import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_dft(img, plot=False):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray_img)
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
    img = cv2.imread("P2/donald_x-ray.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Original image")
    plt.show()
    f_shift = img_dft(img,True)