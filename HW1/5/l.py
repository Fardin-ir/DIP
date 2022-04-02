import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/iran_population_pyramid.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
white = (255,255,255)
black = (13,13,13)
dark_blue = (0,32,96)
dark_red = (115,0,0)

x_start = int(image.shape[1]/2)
dark_blue_idxs = np.where((image == dark_blue).all(axis = 2))
black_idxs = np.where((image[:,:] == black).all(axis = 2))
x_left = np.min(black_idxs[1])
y_start = np.max(dark_blue_idxs[0])
y_end = np.min(black_idxs[0])
width = x_start - x_left
height = y_start - y_end
#image[y_start-10:y_start+10,x_start-10:x_start+10] = (0,32,225)
#image[y_start-10:y_start+10,x_left-10:x_left+10] = (0,32,225)
#image[y_end-10:y_end+10,x_start-10:x_start+10] = (0,32,225)

def find_population(y):
    male = x_start
    female = x_start
    while((image[y,male] != dark_blue).any() and (image[y,male] != white).any()):
        male -= 1
    while((image[y,female] != dark_red).any() and (image[y,female] != white).any()):
        female += 1
    return (x_start-male)/width*1000000, (female-x_start)/width*1000000

male_population = 0
female_population = 0
for i in range(101):
    male, female = find_population(y_start-int(height/101*i)-2)
    male_population += male
    female_population +=female

print("male population:",male_population)
print("female population:",female_population)

plt.imshow(image)
plt.show()