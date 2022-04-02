import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import matplotlib

image = image.imread("P1/checker_shadow_illusion.png")
image = image[:,:,:3]
color = image[200,500]

image[np.all(image != color, axis=-1)] = 1
plt.imshow(image)
plt.show()