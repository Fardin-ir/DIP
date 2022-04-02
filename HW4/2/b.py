import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    img_M = cv2.imread("P2/raisi.png")
    img_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2RGB)
    img_N = cv2.imread("P2/rouhani.png")
    img_N = cv2.cvtColor(img_N, cv2.COLOR_BGR2RGB)

    plt.imshow(img_M)
    P_M = plt.ginput(15, timeout=200)

    plt.imshow(img_N)
    P_N = plt.ginput(15, timeout=200)

    fig, axs = plt.subplots(2)
    axs[0].imshow(img_M)
    axs[1].imshow(img_N)

    for i,set in enumerate([P_M, P_N]):
        for point in [(0,0),(854,0),(0,708),(854,708)]:
            P_M.append(point)
            P_N.append(point)
        for point in set:
            axs[i].scatter(point[0],point[1], c='r', s=10)
    plt.show()

    with open("P_M.txt", "w") as txt_file:
        for point in P_M:
            txt_file.write(str(point[0]) + "," + str(point[1]) + "\n")

    with open("P_N.txt", "w") as txt_file:
        for point in P_N:
            txt_file.write(str(point[0]) + "," + str(point[1]) + "\n")