import numpy as np
import matplotlib.pyplot as plt
import cv2





if __name__ == "__main__":
    text = cv2.imread("P6/III/keep_calm.png")
    text = cv2.cvtColor(text, cv2.COLOR_BGR2RGB)
    shirt = cv2.imread("P6/III/t-shirt.png")
    shirt = cv2.cvtColor(shirt, cv2.COLOR_BGR2RGB)

    mask = np.zeros(shirt.shape)
    h,w = text.shape[0], text.shape[1]
    x_center, y_center = shirt.shape[0]//2, shirt.shape[1]//2
    mask[x_center-h//2:x_center+h//2, y_center-w//2:y_center+w//2] = text//255
    mask[mask != (0,0,0)] = 1
    plt.imsave("P6/III/mask.png", mask)
    mod_text = np.zeros(shirt.shape)
    mod_text = mask
    mod_text[np.where(np.all(mod_text == (0,0,0), axis=-1))] = shirt[500,500]/255
    plt.imsave("P6/III/text.png", mod_text)