import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/provinces_of_iran_by_population.png",cv2.IMREAD_UNCHANGED)
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

color_range_4m = [(0,0,30),(1,1,51)]
mask_4m = cv2.inRange(image,color_range_4m[0],color_range_4m[1])
idxs_4m = np.where(mask_4m == 255)
image[idxs_4m] = (57,57,57)

plt.imshow(image)
plt.show()

countrie_area = image.shape[0] * image.shape[1] - transparent_area
area_4m = len(idxs_4m[0])

print("percentage of provinces with more than 4 million population: ", area_4m/countrie_area*100)

