import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt

image = image.imread("P1/gradient_optical_illusion.png")
color = image[372, 512]
image[np.where(image != color)] = 1
plt.imshow(image,cmap='gray', vmin=0, vmax=1)
plt.show()