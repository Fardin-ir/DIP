import cv2
import matplotlib.pyplot as plt
import numpy as np
from align_imgs import align_imgs
from c import *

if __name__ == "__main__":
    trump1 = cv2.imread("P3/Images/joe_1.png")
    trump2 = cv2.imread("P3/Images/joe_4.png")
    trump1 = cv2.cvtColor(trump1, cv2.COLOR_BGR2GRAY)
    trump2 = cv2.cvtColor(trump2, cv2.COLOR_BGR2GRAY)
    trump1,trump2 = align_imgs(trump1,trump2)
    trump2 = cv2.resize(trump2,(trump1.shape[1],trump1.shape[0]))
    cutoffs = [5, 15, 25, 35, 55]
    for i,cutoff in enumerate(cutoffs):
        filtered_trump1, filtered_trump2 = hybrid_images(trump1,trump2,cutoff,cutoff)
        hybrid_image = filtered_trump1 + filtered_trump2
        plt.subplot(2, 3, i + 1)
        plt.imshow(hybrid_image, cmap="gray")
        plt.title("Hybrid Image")
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    plt.show()
