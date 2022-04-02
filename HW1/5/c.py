import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/internet_censorship_map.png",cv2.IMREAD_UNCHANGED)
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
water_color = (176,224,230)
water_idxs = np.where((image == water_color).all(axis = 2))

selective_color_range = [(237,237,206),(255,255,221)]
selective_mask = cv2.inRange(image,selective_color_range[0],selective_color_range[1])
selective_idxs = np.where(selective_mask == 255)
image[selective_idxs] = (0,32,255)

little_color_range = [(134,220,134),(152,251,152)]
little_mask = cv2.inRange(image,little_color_range[0],little_color_range[1])
little_idxs = np.where(little_mask == 255)
image[little_idxs ] = (0,32,255)

plt.imshow(image)
plt.show()

countries_area = image.shape[0] * image.shape[1] - transparent_area - len(water_idxs[0])
area = len(selective_idxs[0]) + len(little_idxs[0])

print("percentage of countries with selective or little censorship: ", area/countries_area*100)

