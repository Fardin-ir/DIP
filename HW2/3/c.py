import numpy as np
import matplotlib.pyplot as plt
import cv2, glob
from a import import_images


def adaptive_mean_thresholding(img, block, const):
    row = np.ceil(img.shape[0] / block).astype(int)
    col = np.ceil(img.shape[1] / block).astype(int)

    for i in range(row):
        for j in range(col):
            x = [i * block, (i + 1) * block]
            y = [j * block, (j + 1) * block]
            if x[1] > img.shape[0]:
                x[1] = img.shape[0]
            if y[1] > img.shape[1]:
                y[1] = img.shape[1]
            sub_img = img[x[0]:x[1], y[0]:y[1]]
            thresh = np.mean(sub_img) - const
            sub_img[sub_img >= thresh] = 255
            sub_img[sub_img != 255] = 0
    return img

images = import_images()
params = [[10,20],[10,20],[10,20],[40,30]]
for i in range(len(images)):
    block,const = params[i]
    thr_image = adaptive_mean_thresholding(images[i],block,const)
    plt.imshow(thr_image, cmap='gray')
    plt.title(f"block_size={block}, const={const}")
    plt.show()