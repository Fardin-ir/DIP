import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from d import *


if __name__ == "__main__":
    img_M = cv2.imread("P2/raisi.png")
    img_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2RGB)
    img_N = cv2.imread("P2/rouhani.png")
    img_N = cv2.cvtColor(img_N, cv2.COLOR_BGR2RGB)

    P_M = pd.read_csv("P_M.txt", header=None)
    P_N = pd.read_csv("P_N.txt", header=None)
    P_M = P_M.values
    P_N = P_N.values
    height, width, layers = img_M.shape
    video = cv2.VideoWriter('video.avi', 0, 30, (width, height))

    for i in range(61):
        alpha = i/60
        I = final(alpha, img_M, img_N, P_M, P_N)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
        video.write(I)

    cv2.destroyAllWindows()
    video.release()