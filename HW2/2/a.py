import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import cv2

img = cv2.imread('donald_plays_golf_1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


filter = np.array([[-1,-2,-1],
                   [-2,16,-2],
                   [-1,-2,-1]]) / 4

res = signal.convolve2d(img, filter, mode='same')
res = np.clip(res,0,255)
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
plt.figure()
plt.imshow(res, cmap='gray')
plt.show()