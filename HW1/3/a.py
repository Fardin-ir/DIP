import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.metrics.pairwise import euclidean_distances



def find_nearest_k(data,k_arr):
    dist = euclidean_distances(data,k_arr)
    labels = np.argmin(dist,axis=1)
    return labels

def kmeans(data,k,iter):
    k_arr = []
    for i in range(k):
        rand = data[np.random.randint(0,data.shape[0])]
        k_arr.append(rand)
    k_arr = np.array(k_arr)
    lables = np.array([])
    for j in range(iter):
        k_arr_old = k_arr.copy()
        lables = find_nearest_k(data,k_arr)
        for i in range(len(k_arr)):
            k_arr[i,:] = np.mean(data[np.where(lables==i)],axis=0)
        if abs(np.sum(k_arr - k_arr_old)) < 0.0001:
            break
    return k_arr,lables


image = cv2.imread("P3/the_dress.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))

pixel_values = np.float32(pixel_values)

fig, ax = plt.subplots(2,2)
k = [2, 3, 5, 7]
for j in range(len(k)):
    k_arr, labels = kmeans(pixel_values,k[j],100)
    print(k_arr)
    N, bins, patches = ax[int(j/2),j%2].hist(labels, len(k_arr),edgecolor='black', linewidth=1)
    for i in range(len(k_arr)):
        patches[i].set_facecolor(rgb2hex(k_arr[i]/255))
    ax[int(j/2),j%2].set_xticks(np.unique(labels))

plt.show()
