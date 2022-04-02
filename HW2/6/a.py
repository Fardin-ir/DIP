import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

def extract_background(images, numFrames, method=np.average, plot=False):
    # extract background, method = np.average
    backgrounds = []
    for numFrame in numFrames:
        backgrounds.append(method(images[:numFrame], axis=(0)).astype('uint8'))
    if plot:
        plt.figure()
        for i in range(len(backgrounds)):
            plt.subplot(2, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imsave(f"a/n{numFrames[i]}_{method.__name__}.jpg",backgrounds[i])
            plt.imshow(backgrounds[i])
            plt.xlabel(f'numFrame = {numFrames[i]}, method={method.__name__}')
        plt.show()
    return backgrounds

if __name__ == "__main__":
    #import all frames to images array
    filenames = glob.glob("P6/frames/*.png")
    filenames.sort()
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)
    # call function with np.average
    extract_background(images,[2,5,10,20],plot=True,method=np.average)
    # call function with np.median
    extract_background(images, [2, 5, 10, 20], plot=True, method=np.median)



