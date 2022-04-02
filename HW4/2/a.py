import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    img_M = cv2.imread("P2/raisi.png")
    img_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2RGB)/255
    img_N = cv2.imread("P2/rouhani.png")
    img_N = cv2.cvtColor(img_N, cv2.COLOR_BGR2RGB)/255

    for i in range(10):
        alpha = i/9
        I = img_M * (1 - alpha) + img_N * alpha
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(I)
    plt.tight_layout()
    plt.show()

    I = img_M * (1 - 0.5) + img_N * 0.5
    plt.imshow(I)
    plt.show()