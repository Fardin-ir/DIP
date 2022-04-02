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

def find_y_values(x):
    y_rural = max_y_green
    while((image[y_rural,x] == green).all()):
        y_rural -= 1
    y_urban = y_rural - 1
    while((image[y_urban,x] == red).all()):
        y_urban -= 1
    return  max_y_green - y_rural,max_y_green - y_urban

range_1996to2006 = max_x_green - min_x_green
range_1996to2006 = [max_x_green - 2 * int(range_1996to2006/6), max_x_green - int(range_1996to2006/6)]
#image[max_y_green-10:max_y_green+10,range_1996to2006[0]-10:range_1996to2006[0]+10] = (0,32,225)
#image[max_y_green-10:max_y_green+10,range_1996to2006[1]-10:range_1996to2006[1]+10] = (0,32,225)

total = 0
rural = 0
for i in range(range_1996to2006[0],range_1996to2006[1]+1):
    y_rural,y_urban = find_y_values(i)
    total += y_urban
    rural += y_rural

print("ratio between 1996 to 2006: ", rural/total)

plt.imshow(image)
plt.show()