import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/iran_population-urban_vs_rural.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

green = (51,153,51)
red = (255,51,51)
image[np.where((image == (212,8,8)).all(axis = 2))] = red
image[np.where((image == (8,110,8)).all(axis = 2))] = green

green_idxs = np.where((image == green).all(axis = 2))

max_y_green = np.max(green_idxs[0])
min_x_green = np.min(green_idxs[1])
max_x_green = np.max(green_idxs[1])
#image[max_y_green-10:max_y_green+10,min_x_green-10:min_x_green+10] = (0,32,225)
#image[max_y_green-10:max_y_green+10,max_x_green-10:max_x_green+10] = (0,32,225)

def find_y_values(x):
    y_rural = max_y_green
    while((image[y_rural,x] == green).all()):
        y_rural -= 1
    y_urban = y_rural - 1
    while((image[y_urban,x] == red).all()):
        y_urban -= 1
    return  max_y_green - y_rural,max_y_green - y_urban

total = 0
urban = 0
for i in range(min_x_green,max_x_green+1):
    y_rural,y_urban = find_y_values(i)
    total += y_urban
    urban += (y_urban - y_rural)

print("ratio: ", urban/total)

plt.imshow(image)
plt.show()