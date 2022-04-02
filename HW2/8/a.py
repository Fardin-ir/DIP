import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim

def guided_filter(p, I, r, e, same_shape=True):
    '''
    For part e
    In case that p is rgb but I is gray(i.e. they have not same shape, convert
    I to have shape of (...,...,3)
    '''
    if not(same_shape):
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)

    # size of kernel
    r = 2 * r - 1
    # cv2.blur is a mean filter
    mean_I = cv2.blur(I,(r,r))
    mean_p = cv2.blur(p,(r,r))
    corr_I = cv2.blur(np.multiply(I,I),(r,r))
    corr_p = cv2.blur(np.multiply(I,p),(r,r))
    var_I = corr_I - np.multiply(mean_I,mean_I)
    cov_Ip = corr_p - np.multiply(mean_I,mean_p)
    a = np.divide(cov_Ip,(var_I+e))
    b = mean_p - np.multiply(a, mean_I)
    mean_a = cv2.blur(a,(r,r))
    mean_b = cv2.blur(b,(r,r))
    q = np.multiply(mean_a,I) + mean_b
    # In case that same_shape == False, we might get out of bounds a little
    q = np.clip(q, 0, 1)
    return q

def get_noisy_image(img,std):
    noisy_img = img + np.random.normal(0, std, img.shape)
    # might get out of bounds due to noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

if __name__ == '__main__':
    aush = cv2.imread('P8/aush.png')
    # aush/255 because in matplotlib, array of float should be in range [0,1]
    aush = cv2.cvtColor(aush, cv2.COLOR_BGR2GRAY)/255
    # add noise to aush!
    noisy_aush = get_noisy_image(aush,0.1)
    # plot and save noisy image
    plt.imshow(noisy_aush, cmap='gray')
    plt.imsave('a/noisy_aush.jpg',noisy_aush, cmap='gray')
    plt.show()
    # define list of parameters
    r_list = [2,4,8]
    e_list = [0.1**2, 0.2**2, 0.4**2]
    # plot and save result for different set of parameters
    fig, ax = plt.subplots(3, 3)
    for i in range(len(r_list)):
        for j in range(len(e_list)):
            guided_aush = guided_filter(noisy_aush,noisy_aush,r_list[i],e_list[j])
            ax[i, j].imshow(guided_aush, cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            # show parameters as x-label
            ax[i, j].set_xlabel(f'r={r_list[i]}, e={e_list[j]:.2f}')
            # calculate and display ssim and psnr as images title
            psnr = cv2.PSNR(aush, guided_aush)
            ssim = compare_ssim(aush, guided_aush)
            ax[i, j].set_title(f'psnr={psnr:.2f}, ssim={ssim:.2f}')
            # save images with appropriate name
            plt.imsave(f'a/g_r={r_list[i]},e={e_list[j]:.2f}.jpg',guided_aush, cmap='gray')
    plt.show()