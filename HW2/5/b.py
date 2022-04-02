
import numpy as np
import matplotlib.pyplot as plt
import cv2, glob
from a import *

def enh(img,tile):
    row = np.ceil(img.shape[0] / tile).astype(int)
    col = np.ceil(img.shape[1] / tile).astype(int)
    new = img.copy()
    for i in range(row):
        x = [i * tile - 2, i * tile + 2]
        if x[1] > new.shape[0]:
            x[1] = new.shape[0]
        if x[0] < 0:
            x[0] = 0
        new[x[0]:x[1],:] = cv2.blur(new[x[0]:x[1],:],(5,5))
    for i in range(col):
        y = [i * tile - 2, i * tile + 2]
        if y[1] > new.shape[1]:
            y[1] = new.shape[1]
        if y[0] < 0:
            y[0] = 0
        new[:,y[0]:y[1]] = cv2.blur(new[:,y[0]:y[1]], (5, 5))

    return new

if __name__ == "__main__":
    images = import_images(plot=True)
    for image in images:
        new_img = adaptive_hist_eq(image,200,hist_eq)
        new_img = enh(new_img,200)
        plt.figure()
        plt.imshow(new_img, cmap='gray')
        plt.show()