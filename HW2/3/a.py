import numpy as np
import matplotlib.pyplot as plt
import cv2, glob

def import_images(plot=False):
    filenames = glob.glob("P3/*.png")
    filenames.sort()
    print(filenames)
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    images = np.array(images)
    if plot:
        plt.figure()
        for i in range(len(images)):
            plt.subplot(2, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i],cmap='gray')
        plt.show()
    return images

def global_thresholding(images, thrs):
    for i in range(len(images)):
        thr_image = images[i]
        thr_image[thr_image >= thrs[i]] = 255
        thr_image[thr_image != 255] = 0
        plt.imshow(thr_image,cmap='gray')
        plt.show()

if __name__ == "__main__":
    images = import_images(plot=True)
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.grid(False)
        plt.hist(images[i].ravel(),40)
    plt.show()
    global_thresholding(images,[95,100,160,50])


