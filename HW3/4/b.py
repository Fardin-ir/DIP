import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from a import *

def recog_rate(H, test_images,trh):
    recog_arr = []
    for test_image in test_images:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_dft = np.fft.fft2(test_image)
        output = np.fft.ifft2(test_dft * np.conj(H))
        output = np.real(np.fft.ifftshift(output))
        recog_arr.append(recog(output,trh))
    print(recog_arr)
    recog_arr = np.array(recog_arr)
    recog_arr = np.sum(recog_arr[recog_arr == True]) / len(recog_arr)
    return recog_arr

def import_test_images(class_index, postfix):
    test_images = []
    if class_index <= 50:
        id = class_index
        prefix = "M"
    else:
        id = class_index - 50
        prefix = "W"
    X = []
    for filename in glob.glob(f"P4/test2/*.bmp"):
        if filename.split("-")[-1].split(".")[0] not in postfix:
            continue
        if int(filename.split("-")[1]) != id:
            continue
        if filename.split("-")[0].split("\\")[1] != prefix:
            continue
        print(filename)
        img = cv2.imread(filename)
        test_images.append(img)
    return test_images

if __name__ == "__main__":
    H_arr = np.zeros((100,165,120), dtype=complex)
    for filename in glob.glob("a/*.csv"):
        id = int(filename.split("_")[1].split(".")[0])
        H = pd.read_csv(filename, header=None)
        H = H.values
        complex_out = []
        for row in H:
            comp_row = [complex(x) for x in row]
            complex_out.append(comp_row)
        H = np.array(complex_out)
        H_arr[id - 1] = H
    print(H_arr.shape)

    postfix = [str(x) for x in range(14,21)]

    recognition_rates = []
    for i in range(1,101):
        recognition_rates.append(recog_rate(H_arr[i-1],import_test_images(i,postfix),5))
    recognition_rates = np.array(recognition_rates)

    plt.plot(recognition_rates)
    plt.title(f"Mean recognition rate: {np.mean(recognition_rates):.2f}")
    plt.show()