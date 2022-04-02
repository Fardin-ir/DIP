import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from c import extract_foreground_mask

def mask_image(mask,image):
    image = image.copy()
    image[np.where(mask == 0)] = (0,0,0)
    print(mask)
    return image

def extract_foreground_players(images, numFrames, test_fram, method=np.average, plot=False):
    foreground_masks = extract_foreground_mask(images,[2,5,10,20],test_frame,method=method)
    foreground_players = []
    for foreground_mask in  foreground_masks:
        foreground_players.append(mask_image(foreground_mask,test_fram))
    if plot:
        plt.figure()
        for i in range(len(foreground_players)):
            plt.subplot(2, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imsave(f"d/n{numFrames[i]}_method={method.__name__}.jpg", foreground_players[i])
            plt.imshow(foreground_players[i])
            plt.xlabel(f'numFrame = {numFrames[i]},method={method.__name__}.jpg"')
        plt.show()
    return foreground_players

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
    # import test frame
    test_frame = cv2.imread('P6/test/football_test.png')
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    # call function np.average
    extract_foreground_players(images,[2,5,10,20],test_frame,plot=True,method=np.average)
    # call function np.median
    extract_foreground_players(images, [2, 5, 10, 20], test_frame, plot=True, method=np.median)