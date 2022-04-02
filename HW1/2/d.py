import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
from a import extract_channels
from b import stack_channels
import cv2
import time


def corrcoef(x,y):
    return np.corrcoef(x.ravel(), y.ravel())

def align_two_matrix(mat1,mat2,low_res_search):
    original_mat1 = mat1 = mat1.copy()
    original_mat2 = mat2 = mat2.copy()
    if(low_res_search):
        mat1 = cv2.resize(mat1, dsize=(int(mat1.shape[1]/2),int(mat1.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
        mat2 = cv2.resize(mat2, dsize=(int(mat2.shape[1]/2), int(mat2.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    h = 0
    v = 0
    min_similarity = 0
    for i in range(-15,+15):
        for j in range(-15,15):
            rolled_mat2 = np.roll(np.roll(mat2,i,axis=0),j,axis=1)
            similarity = np.sum(corrcoef(mat1, rolled_mat2))
            if similarity>min_similarity:
                min_similarity = similarity
                h = i
                v = j
    if low_res_search:
        return np.roll(np.roll(original_mat2, h * 2, axis=0), v * 2, axis=1)
    else:
        return np.roll(np.roll(original_mat2, h, axis=0), v, axis=1)

def align_results(r,g,b):
    r = align_two_matrix(g,r,True)
    b = align_two_matrix(g,b,True)
    r = align_two_matrix(g, r, False)
    b = align_two_matrix(g, b, False)
    return stack_channels(r,g,b)


if __name__ == "__main__":
    start_time = time.time()

    fig, ax = plt.subplots(2, 2)
    image = img.imread("P2/01.jpg")
    b, g, r = extract_channels(image)
    rgb = align_results(r, g, b)
    ax[0, 0].imshow(rgb)
    image = img.imread("P2/02.jpg")
    b, g, r = extract_channels(image)
    rgb = align_results(r, g, b)
    ax[0, 1].imshow(rgb)
    image = img.imread("P2/03.jpg")
    b, g, r = extract_channels(image)
    rgb = align_results(r, g, b)
    ax[1, 0].imshow(rgb)
    image = img.imread("P2/04.jpg")
    b, g, r = extract_channels(image)
    rgb = align_results(r, g, b)
    ax[1, 1].imshow(rgb)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

