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


def gaussian_kernel(img, sigma_x, sigma_y):
    dimension_x = img.shape[0]
    dimension_y = img.shape[1]
    x = cv2.getGaussianKernel(dimension_x, sigma_x)
    y = cv2.getGaussianKernel(dimension_y, sigma_y)
    kernel = x.dot(y.T)
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    return kernel

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

if __name__ == "__main__":
    img1 = cv2.imread("P2/donald_childhood.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.imread("P2/donald_graduation.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


    trh = 40
    q = 50
    for c in range(3):
        dft, dft_image = get_channels_dft(img1)
        R = dft_image[:,:,c]
        R = cv2.blur(R, (20,20))
        R = (R - R.min()) / (R.max() - R.min()) * 255
        R[R > trh] = 0
        R[R != 0] = 1
        R = R.astype(np.float)
        R += gaussian_kernel(R, q, q)
        R = np.clip(R,0,1)
        highpass = np.ones(R.shape)
        highpass[50:R.shape[0]-50,50:R.shape[1]-50] = 0
        R[np.where(highpass==1)] = 0
        plt.subplot(2,3, c + 1)
        plt.imshow(R, cmap="gray")
        img1[:, :, c] = img_idft(dft[:, :, c]*R,False)
        plt.subplot(2, 3, c +4)
        plt.imshow(img1[:, :, c], cmap="gray")
    plt.tight_layout()
    plt.show()

    plt.imshow(img1)
    plt.show()