import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

def img_dft(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    f_abs = np.abs(f_shift) + 1
    f_bounded = 20 * np.log10(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    return f_shift,f_img

def recog(correlation,trh):
    mid_x, mid_y = correlation.shape[0]//2, correlation.shape[1]//2
    peak = np.max(correlation[mid_x-2:mid_x+2, mid_y-2:mid_y+2])
    sidelobe = correlation[mid_x-10:mid_x+10, mid_y-10:mid_y+10]
    sidelobe[10-2:10+2,10-2:10+2] = -10
    mean = np.mean(sidelobe[sidelobe != -10])
    std = np.std(sidelobe[sidelobe != -10])
    return ((peak-mean)/std) > trh



if __name__ == "__main__":
    H_arr = np.zeros((100,165,120), dtype=complex)
    for filename in glob.glob("c/*.csv"):
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

    test_image = cv2.imread("P4/test2/M-001-14.bmp")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_dft = np.fft.fft2(test_image)

    output = np.fft.ifft2(test_dft * np.conj(H_arr[0]))
    output = np.real(np.fft.ifftshift(output))
    for i in range(output.shape[1]):
        plt.plot(output[:,i])
    plt.show()

    plt.imshow(np.abs(H_arr[0]), cmap="gray")
    plt.show()

    print(recog(output,5))