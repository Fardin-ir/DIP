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

def get_channels_dft(img):
    channels_dft = []
    channels_dft_img = []
    for c in range(img.shape[2]):
        dft, dft_image = img_dft(img[:,:,c])
        channels_dft.append(dft)
        channels_dft_img.append(dft_image)
    dft = np.dstack([channels_dft[0],
                        channels_dft[1],
                         channels_dft[2]])
    dft_image = np.dstack([channels_dft_img[0],
                            channels_dft_img[1],
                            channels_dft_img[2]])
    return dft,dft_image



if __name__ == "__main__":
    img1 = cv2.imread("P2/donald_childhood.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("P2/donald_graduation.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    for img in [img1, img2]:
        dft, dft_image = get_channels_dft(img)
        c_names = ["R", "G", "B"]
        for c, name in enumerate(c_names):
            plt.subplot(2, 3, c + 1)
            plt.imshow(img[:,:,c], cmap="gray")
            plt.title(f"Channel {name}")
            plt.subplot(2, 3, c+4 )
            plt.imshow(dft_image[:,:,c], cmap="gray")
        plt.tight_layout(pad=1, w_pad=0, h_pad=0)
        plt.show()

    for img in [img1, img2]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        dft, dft_image = get_channels_dft(img)
        c_names = ["H", "S", "V"]
        for c, name in enumerate(c_names):
            plt.subplot(2, 3, c + 1)
            plt.imshow(img[:,:,c], cmap="gray")
            plt.title(f"Channel {name}")
            plt.subplot(2, 3, c+4 )
            plt.imshow(dft_image[:,:,c], cmap="gray")
        plt.tight_layout(pad=1, w_pad=0, h_pad=0)
        plt.show()

    for img in [img1, img2]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        dft, dft_image = get_channels_dft(img)
        c_names = ["L", "A", "B"]
        for c, name in enumerate(c_names):
            plt.subplot(2, 3, c + 1)
            plt.imshow(img[:,:,c], cmap="gray")
            plt.title(f"Channel {name}")
            plt.subplot(2, 3, c+4 )
            plt.imshow(dft_image[:,:,c], cmap="gray")
        plt.tight_layout(pad=1, w_pad=0, h_pad=0)
        plt.show()

    for img in [img1, img2]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        dft, dft_image = get_channels_dft(img)
        c_names = ["Y", "CB", "CR"]
        for c, name in enumerate(c_names):
            plt.subplot(2, 3, c + 1)
            plt.imshow(img[:,:,c], cmap="gray")
            plt.title(f"Channel {name}")
            plt.subplot(2, 3, c+4 )
            plt.imshow(dft_image[:,:,c], cmap="gray")
        plt.tight_layout(pad=1, w_pad=0, h_pad=0)
        plt.show()


