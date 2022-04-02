import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt

def extract_channels(img):
    h = img.shape[0]
    b = img[:int(h/3),:]
    g = img[int(h/3):int(2*h/3),:]
    r = img[int(2*h/3):h-1,:]
    return b,g,r

def stack_channels(r,g,b):
    return np.dstack((r, g, b))

def corrcoef(x,y):
    return np.corrcoef(x.ravel(), y.ravel())

def align_two_matrix(mat1,mat2):
    h = 0
    v = 0
    min_similarity = 0
    for i in range(-30,+30):
        for j in range(-30,30):
            rolled_mat2 = np.roll(np.roll(mat2,i,axis=0),j,axis=1)
            similarity = np.sum(corrcoef(mat1, rolled_mat2))
            if similarity>min_similarity:
                min_similarity = similarity
                h = i
                v = j
    return np.roll(np.roll(mat2,h,axis=0),v,axis=1)

def align_results(r,g,b):
    r = align_two_matrix(g,r)
    b = align_two_matrix(g,b)
    return stack_channels(r,g,b)

image = image.imread("P2/04.jpg")
b,g,r = extract_channels(image)

rgb = stack_channels(r,g,b)
plt.imshow(rgb)
plt.show()

rgb = align_results(r,g,b)
plt.imshow(rgb)
plt.show()