import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import matplotlib

image = image.imread("P1/ebbinghaus_illusion.png")
print(image.shape)
image = image[:,:,:3]
pixel_values = image.reshape((-1, 3))
color = image[200,200]
image[np.all(image != color, axis=-1)] = 1
plt.imshow(image)
plt.show()