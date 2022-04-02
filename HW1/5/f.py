import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/provinces_of_iran_by_population.png",cv2.IMREAD_UNCHANGED)
image = image[:,int(image.shape[1]/2):]
transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

color_range_1m = [(177,177,221),(204,204,255)]
mask_1m = cv2.inRange(image,color_range_1m[0],color_range_1m[1])
idxs_1m = np.where(mask_1m == 255)
image[idxs_1m] = (57,57,57)

plt.imshow(image)
plt.show()

countrie_area = image.shape[0] * image.shape[1] - transparent_area
area_1m = len(idxs_1m[0])

print("percentage of provinces with less than 1 million population in the eastern part: ", area_1m/countrie_area*100)

