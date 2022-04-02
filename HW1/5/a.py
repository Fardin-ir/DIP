import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/internet_censorship_map.png",cv2.IMREAD_UNCHANGED)
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
water_color = (176,224,230)

water_idxs = np.where((image == water_color).all(axis = 2))
image[water_idxs] = (100,100,100)

plt.imshow(image)
plt.show()

percentage = len(water_idxs[0])/(image.shape[0] * image.shape[1] - transparent_area) * 100

print(" percentage of water on earth:", percentage)

