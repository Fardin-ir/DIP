from matplotlib import image as img
from matplotlib import pyplot as plt


def extract_channels(img):
    h = img.shape[0]
    b = img[:int(h/3),:]
    g = img[int(h/3):int(2*h/3),:]
    r = img[int(2*h/3):h-1,:]
    return b,g,r

if __name__ == "__main__":
    image = img.imread("P2/01.jpg")
    b,g,r = extract_channels(image)
    fig, ax = plt.subplots(4,3)
    ax[0,0].imshow(b, cmap="gray")
    ax[0,1].imshow(g, cmap="gray")
    ax[0,2].imshow(r, cmap="gray")
    ax[0,0].set_title("b")
    ax[0,1].set_title("g")
    ax[0,2].set_title("r")

    image = img.imread("P2/02.jpg")
    b, g, r = extract_channels(image)
    ax[1, 0].imshow(b, cmap="gray")
    ax[1, 1].imshow(g, cmap="gray")
    ax[1, 2].imshow(r, cmap="gray")

    image = img.imread("P2/03.jpg")
    b, g, r = extract_channels(image)
    ax[2, 0].imshow(b, cmap="gray")
    ax[2, 1].imshow(g, cmap="gray")
    ax[2, 2].imshow(r, cmap="gray")

    image = img.imread("P2/04.jpg")
    b, g, r = extract_channels(image)
    ax[3, 0].imshow(b, cmap="gray")
    ax[3, 1].imshow(g, cmap="gray")
    ax[3, 2].imshow(r, cmap="gray")
    plt.show()