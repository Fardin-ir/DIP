import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.spatial import Delaunay

def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img_M, img_N, imgMorph, tri_M, tri_N, t, alpha):
    # Find bounding rectangle for each triangle
    r_M = cv2.boundingRect(np.float32([tri_M]))
    r_N = cv2.boundingRect(np.float32([tri_N]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    tM_Rect = []
    tN_Rect = []
    t_Rect = []

    for i in range(0, 3):
        t_Rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        tM_Rect.append(((tri_M[i][0] - r_M[0]), (tri_M[i][1] - r_M[1])))
        tN_Rect.append(((tri_N[i][0] - r_N[0]), (tri_N[i][1] - r_N[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    imgM_Rect = img_M[r_M[1]:r_M[1] + r_M[3], r_M[0]:r_M[0] + r_M[2]]
    imgN_Rect = img_N[r_N[1]:r_N[1] + r_N[3], r_N[0]:r_N[0] + r_N[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(imgM_Rect, tM_Rect, t_Rect, size)
    warpImage2 = applyAffineTransform(imgN_Rect, tN_Rect, t_Rect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    imgMorph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = imgMorph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def final(alpha, img_M, img_N, P_M, P_N):
    P_I = (1 - alpha) * P_M + alpha * P_N

    img_M = np.float32(img_M)
    img_N = np.float32(img_N)

    # Allocate space for final output
    imgMorph = np.zeros(img_M.shape, dtype=img_M.dtype)

    # Read triangles from tri.txt
    tri_I = Delaunay(P_I)
    for i in range(len(tri_I.simplices)):
        x, y, z = tri_I.simplices[i]

        x = int(x)
        y = int(y)
        z = int(z)

        tri_M = [P_M[x], P_M[y], P_M[z]]
        tri_N = [P_N[x], P_N[y], P_N[z]]
        t = [P_I[x], P_I[y], P_I[z]]

        # Morph one triangle at a time.
        morphTriangle(img_M, img_N, imgMorph, tri_M, tri_N, t, alpha)

    return np.uint8(imgMorph)


if __name__ == "__main__":
    img_M = cv2.imread("P2/raisi.png")
    img_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2RGB)
    img_N = cv2.imread("P2/rouhani.png")
    img_N = cv2.cvtColor(img_N, cv2.COLOR_BGR2RGB)

    P_M = pd.read_csv("P_M.txt", header=None)
    P_N = pd.read_csv("P_N.txt", header=None)
    P_M = P_M.values
    P_N = P_N.values

    for i in range(10):
        alpha = i/9
        I = final(alpha,img_M,img_N,P_M,P_N)
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(I)
    plt.tight_layout()
    plt.show()

    I = final(0.5,img_M,img_N,P_M,P_N)
    plt.imshow(I)
    plt.show()