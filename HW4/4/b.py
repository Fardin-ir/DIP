import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage, misc



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
    img = cv2.imread("P4/post-mortem_2.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.equalizeHist(img)/255
    img = wgif(img, img, 5, 5, 0.2 ** 2)
    plt.imshow(img,cmap="gray")
    plt.show()

