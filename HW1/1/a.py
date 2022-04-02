import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt

image = image.imread("P1/color_saturation_illusion.png")
bg1 = image[0, 0]
bg2 = image[599, 1199]

image[np.all(image == bg1, axis=-1)] = bg2

plt.imshow(image)
plt.show()

image[np.all(image == bg2, axis=-1)] = 1

plt.imshow(image)
plt.show()