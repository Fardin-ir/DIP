import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
from a import extract_channels

def stack_channels(r,g,b):
    return np.dstack((r, g, b))

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    image = img.imread("P2/01.jpg")
    b,g,r = extract_channels(image)
    rgb = stack_channels(r,g,b)
    ax[0, 0].imshow(rgb)

    image = img.imread("P2/02.jpg")
    b, g, r = extract_channels(image)
    rgb = stack_channels(r, g, b)
    ax[0, 1].imshow(rgb)

    image = img.imread("P2/03.jpg")
    b, g, r = extract_channels(image)
    rgb = stack_channels(r, g, b)
    ax[1, 0].imshow(rgb)

    image = img.imread("P2/04.jpg")
    b, g, r = extract_channels(image)
    rgb = stack_channels(r, g, b)
    ax[1, 1].imshow(rgb)
    plt.show()