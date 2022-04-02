import numpy as np
import matplotlib.pyplot as plt
import cv2, glob
from a import import_images

def otsu(img):
    his, bins = np.histogram(img, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        q1 = np.sum(his[:t]) / (img.shape[0] * img.shape[1])
        q2 = np.sum(his[t:]) / (img.shape[0] * img.shape[1])
        mu1 = np.mean(his[:t])
        mu2 = np.mean(his[t:])
        value = q1 * q2 * (mu1 - mu2) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    img = img.copy()
    print(final_thresh)
    img[img > final_thresh] = 255
    img[img != 255] = 0
    return img,final_thresh

images = import_images()
for i in range(len(images)):
    thr_image, thresh = otsu(images[i])
    plt.imshow(thr_image, cmap='gray')
    plt.title(f"thresh={thresh}")
    plt.show()