
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

grid_idxs = np.where((image == (38,38,38)).all(axis = 2))
print(grid_idxs)

top_left_grid = [60,136]
#image[60-10:60+10,136-10:136+10] = (0,32,225)

def find_population_values(x):
    chart_h = max_y_green - top_left_grid[0]
    y_rural = max_y_green
    while((image[y_rural,x] == green).all()):
        y_rural -= 1
    y_urban = y_rural - 1
    while((image[y_urban,x] == red).all()):
        y_urban -= 1
    return  (max_y_green - y_rural)/chart_h*90,(max_y_green - y_urban)/chart_h*90

range_1976to1986 = max_x_green - min_x_green
range_1976to1986 = [min_x_green + 2 * int(range_1976to1986/6), min_x_green + 3 * int(range_1976to1986/6)]
print(range_1976to1986)
#image[max_y_green-10:max_y_green+10,range_1976to1986[0]-10:range_1976to1986[0]+10] = (0,32,225)
#image[max_y_green-10:max_y_green+10,range_1976to1986[1]-10:range_1976to1986[1]+10] = (0,32,225)

decade = range_1976to1986[1] - range_1976to1986[0]
total = 0
for i in range(10):
    rural, urban = find_population_values(range_1976to1986[0] + i * int(decade/10))
    urban = urban - rural
    print("population in ", 1976 + i, ": " , urban)
    total += urban
print("total urban population in 1976:1986 :", total)

plt.imshow(image)
plt.show()