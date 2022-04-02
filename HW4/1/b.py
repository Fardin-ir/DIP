import numpy as np
import matplotlib.pyplot as plt
import cv2
from a import G_mean

def custom_median(img, kernel_size):
    G_mean_img = img.copy()
    k = int((kernel_size - 1) / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i < k or i > (img.shape[0] - k - 1) or j < k or j > (img.shape[1] - k - 1):
                G_mean_img[i][j] = img[i][j]
            elif img[i,j] in range(0,10) or img[i,j] in range(240,256):
                G_mean_img[i][j] = np.median(img[i-k:i+k,j-k:j+k])

    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img


if __name__ == "__main__":
    img = cv2.imread("P1/nasir_and_dentist.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    area = plt.ginput(2)
    area = np.asarray(area).astype("int")
    plt.show()
    area = img[area[0, 1]:area[1, 1], area[0, 0]:area[1, 0]]
    plt.imshow(area, cmap="gray")
    plt.show()
    plt.hist(area.ravel(), 256, [0, 256])
    plt.show()
    img = custom_median(img,5)
    img = G_mean(img,3)
    plt.imshow(img, cmap="gray")
    plt.show()