
import numpy as np
import matplotlib.pyplot as plt
import cv2, glob

def import_images(plot=False):
    filenames = glob.glob("P5/*.png")
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

def hist_eq(img):
    pdf = (np.histogram(img.ravel(), 256, [0, 255])[0])
    cdf = np.cumsum(pdf)

    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()

    cdf = nj / N

    cdf = cdf.astype('uint8')
    new = cdf[img.ravel()]
    new = new.reshape((img.shape[0], img.shape[1]))

    return new


def adaptive_hist_eq(img, tile, method):
    row = np.ceil(img.shape[0] / tile).astype(int)
    col = np.ceil(img.shape[1] / tile).astype(int)
    new = img.copy()
    for i in range(row):
        for j in range(col):

            x = [i * tile, (i + 1) * tile]
            y = [j * tile, (j + 1) * tile]

            if x[1] > new.shape[0]:
                x[1] = new.shape[0]
            if y[1] > new.shape[1]:
                y[1] = new.shape[1]
            new[x[0]:x[1], y[0]:y[1]] = method(new[x[0]:x[1], y[0]:y[1]])

    return new

if __name__ == "__main__":
    images = import_images(plot=True)
    for images in images:
        new_img = adaptive_hist_eq(images,200,hist_eq)
        plt.figure()
        plt.imshow(new_img, cmap='gray')
        plt.show()
