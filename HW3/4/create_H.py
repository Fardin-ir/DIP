import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd


def create_H_matrix(postfix):
    H_arr = []
    for prefix in ["M", "W"]:
        for id in range(1,51):
            size = 0
            X = []
            for filename in glob.glob("P4/test2/*.bmp"):
                if filename.split("-")[-1].split(".")[0] not in postfix:
                    continue
                if int(filename.split("-")[1]) != id:
                    continue
                if filename.split("-")[0].split("\\")[1] != prefix:
                    continue
                print(filename)
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = img.shape
                dft = np.fft.fft2(img)
                X.append(dft.ravel())
            X = np.array(X).T
            if prefix == "W":
                index = id + 50
            else:
                index = id
            print(index)
            ASD = np.abs(X) ** 2
            sum = np.sum(ASD,axis=1)
            D = sum / X.shape[1]
            d_inv = 1 / D
            D = np.diag(D)
            d_inv = np.diag(d_inv)
            U = np.ones((X.shape[1],1))
            C = X.conj().T
            F = np.linalg.inv(C.dot(d_inv).dot(X))
            Y = F.dot(U)
            E = d_inv.dot(X)
            H = E.dot(Y)
            H = H.reshape((size[0],size[1]))
            H_arr.append(H)
            H = pd.DataFrame(H)
            H.to_csv(f"a/H_{index}.csv", header=None, index=None)
    return np.array(H_arr)

if __name__ == "__main__":
    postfix = ["01", "02", "03", "04", "05", "06", "07"]
    H_arr = create_H_matrix(postfix)
    print(H_arr.shape)
