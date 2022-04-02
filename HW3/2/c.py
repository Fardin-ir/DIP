import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    img1 = cv2.imread("P2/donald_childhood.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("P2/donald_graduation.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    for img in [img1, img2]:
        img_blur = cv2.GaussianBlur(img, (7, 7),20)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(img_blur)
        ax1.set_title("Original Image")
        ax2.set_title("Blured Image")
        fig.tight_layout()
        plt.show()