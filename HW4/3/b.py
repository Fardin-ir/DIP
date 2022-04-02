import itertools
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from a import *

if __name__ == '__main__':
    import glob
    image_list = []
    for filename in glob.glob('P3/II/*.png'):
        print(filename)
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_list.append(im)

    res = main(image_list)

    plt.imshow(res)
    plt.show()