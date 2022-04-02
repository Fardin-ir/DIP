import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2


image = cv2.imread("P4/wheres_wally_1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def block_mask(mask,h,v):
    b_mask = mask.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 255:
                if i-v > 0 & i+v < mask.shape[0] & j-h > 0 & j+h < mask.shape[1]:
                    b_mask[i-v:i+v,j-h:j+h] = 255
    return b_mask


red1 = (225,70,70)
red2 = (245,90,90)
#white1 =  (254,254,254)
#white2 = (255,255,255)
mask_red = cv2.inRange(image, red1, red2)
#mask_white = cv2.inRange(image, white1, white2)
#mask = mask_red+mask_white
mask = mask_red
mask = block_mask(mask,40,40)
result = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(result)
plt.show()
plt.imsave('image1.jpg',arr=result)

