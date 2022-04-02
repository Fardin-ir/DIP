import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/internet_censorship_map.png",cv2.IMREAD_UNCHANGED)
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
water_color = (176,224,230)
water_idxs = np.where((image == water_color).all(axis = 2))

pervasive_color_range = [(178,115,160),(255,153,221)]
pervasive_mask = cv2.inRange(image,pervasive_color_range[0],pervasive_color_range[1])
pervasive_idxs = np.where(pervasive_mask == 255)
image[pervasive_idxs] = (0,32,255)

plt.imshow(image)
plt.show()

countries_area = image.shape[0] * image.shape[1] - transparent_area - len(water_idxs[0])
pervasive_area = len(pervasive_idxs[0])

print("percentage of countries with pervasive censorship: ", pervasive_area/countries_area*100)

