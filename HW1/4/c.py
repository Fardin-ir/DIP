import numpy as np
from matplotlib import pyplot as plt
import cv2
import multiprocessing as mp
import scipy.spatial as sp
strid = 5
image = cv2.imread("image2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original_image = cv2.imread("P4/wheres_wally_2.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

temps = []

temp = cv2.imread("P4/wally_2.jpg")
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

temp[np.where((temp == (255,255,255)).all(axis = 2))] = (0,0,0)


temps.append(temp)
width = temp.shape[1]
height = temp.shape[0]
'''
temp = cv2.imread("P4/wally_4.jpg")
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
temp[np.where((temp == (255,255,255)).all(axis = 2))] = (0,0,0)
temps.append(temp)

temp = cv2.imread("P4/wally_4.jpg")
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
temp[np.where((temp == (255,255,255)).all(axis = 2))] = (0,0,0)
temps.append(temp)
'''

def corrcoef(x,y,z,i):
    s = np.sum(np.corrcoef(x.ravel(), y.ravel()))
    z[i] = s


def siml(img,temps,i,j):
    sum = 0
    mfactor = 1
    factors = [0.8,1]
    for temp in temps:
        sims = mp.Array('d',len(factors))
        process = []
        for i in range(len(factors)):
            resized_temp = resize(temp,factors[i])
            width = resized_temp.shape[1]
            height = resized_temp.shape[0]
            p = mp.Process(target=corrcoef,args = (img[i:i + height, j:j + width], resized_temp,sims,i))
            p.start()
            process.append(p)
        for proces in process:
            proces.join()
        max_sim = np.frombuffer(sims.get_obj()).max()
        sum =+ max_sim
    return sum,mfactor

def resize(img,factor):
    small = cv2.resize(img, (0, 0), fx=factor, fy=factor)
    return small



if __name__ == '__main__':
    map = np.zeros(image.shape)

    idxs = np.where((image != (0,0,0)).any(axis = 2))


    max = 0
    max_idx = []
    max_factor = []
    for k in range(0,len(idxs[0]),strid):
        i = idxs[0][k]
        j = idxs[1][k]
        print(i,j)
        if i > image.shape[0] - 175 or j > image.shape[1] - 175:
            continue
        sim,factor = siml(image, temps,i,j)
        map[i,j] = sim
        if sim > max:
            max = sim
            max_idx.append([i,j])
            max_factor.append(factor)
            print(sim, i,j)

    point = max_idx[-1:][0]
    factor = max_factor[-1:][0]
    x = point[1]
    y = point[0]

    cv2.rectangle(original_image, (x, y), (x+int(width*factor), y+int(height*factor)), (255, 0, 0), 2)

    plt.imshow(original_image)
    plt.imsave("c2.jpg",original_image)
    plt.show()
    map = (np.nan_to_num(map))
    sec = np.amin(np.array(map)[map != np.amin(map)])
    map = ((map-sec)/(map.max()-sec)*255).astype(np.uint8)
    plt.matshow(map)
    plt.show()
