import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.spatial import Delaunay

if __name__ == "__main__":
    img_M = cv2.imread("P2/raisi.png")
    img_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2RGB)
    img_N = cv2.imread("P2/rouhani.png")
    img_N = cv2.cvtColor(img_N, cv2.COLOR_BGR2RGB)

    P_M = pd.read_csv("P_M.txt", header=None)
    P_N = pd.read_csv("P_N.txt", header=None)
    P_M = P_M.values.astype(int)
    P_N = P_N.values.astype(int)
    print(P_M.shape)
    print(P_N.shape)
    tri_M = Delaunay(P_M)
    tri_N = Delaunay(P_N)
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_M)
    axs[0].triplot(P_M[:, 0], P_N[:, 1], tri_M.simplices)
    axs[1].imshow(img_N)
    axs[1].triplot(P_N[:, 0], P_N[:, 1], tri_N.simplices)
    plt.show()