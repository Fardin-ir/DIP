import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/provinces_of_iran_by_population.png",cv2.IMREAD_UNCHANGED)
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

color_range_2m = [(0,0,51),(102,102,204)]
mask_2m = cv2.inRange(image,color_range_2m[0],color_range_2m[1])
idxs_2m = np.where(mask_2m == 255)
image[idxs_2m] = (57,57,57)

plt.imshow(image)
plt.show()

countrie_area = image.shape[0] * image.shape[1] - transparent_area
area_2m = len(idxs_2m[0])

print("percentage of provinces with more than 2 million population: ", area_2m/countrie_area*100)

