import numpy as np
import matplotlib.pyplot as plt
import cv2

def median(img, kernel_size):
    G_mean_img = img.copy()
    k = int((kernel_size - 1) / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i < k or i > (img.shape[0] - k - 1) or j < k or j > (img.shape[1] - k - 1):
                G_mean_img[i][j] = img[i][j]
            else:
                G_mean_img[i][j] = np.median(img[i-k:i+k,j-k:j+k])

    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img


if __name__ == "__main__":
    img = cv2.imread("P1/nasir_smoking_hookah.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = median(img, 5)
    plt.imshow(img, cmap="gray")
    plt.show()