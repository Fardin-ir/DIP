import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("P5/provinces_of_iran_by_population.png",cv2.IMREAD_UNCHANGED)

transparent_area = len(np.where(image[:,:,3] == 0)[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_n = image[:int(image.shape[0]/2),:]
image_s = image[int(image.shape[0]/2):,:]

colors = [(0,0,51),(0,0,102),(51,51,153),(102,102,204),(153,153,204),(220,220,255)]
p = [5,4,3,2,1.5,1]

countrie_area = image.shape[0] * image.shape[1] - transparent_area

population = []
for j in [image_n,image_s]:
    for i in range(len(colors)):
        mask = cv2.inRange(j,colors[i],colors[i])
        idxs = np.where(mask == 255)
        area = len(idxs[0])
        population.append(area/countrie_area*100*p[i])

population = np.asarray(population)
population_n = np.sum(population[:6])
population_s = np.sum(population[6:])

if population_n>population_s:
    print("North has more population")
else:
    print("South has more population")
