from scipy.ndimage import gaussian_filter, rotate
from utils import image_derivatives, normal2chan

import matplotlib.pyplot as plt
import numpy as np

def detect_corners(img: np.ndarray, window: int = 5, threshold: float = 1e-5) -> tuple:
    img = normal2chan(img)
    Ix, Iy = image_derivatives(img)

    A = gaussian_filter(Ix ** 2, sigma=1)
    B = gaussian_filter(Ix * Iy, sigma=1)
    C = gaussian_filter(Iy ** 2, sigma=1)
    H = (A * C - B**2) - 0.04 * (A + C) ** 2

    window = 5
    h, w = img.shape
    r, c = [], []
    for x in range(window, h - window - 1):
        for y in range(window, w - window - 1):
            p = H[x, y]
            region = H[x-window:x+window, y-window:y+window]
            if p > threshold and region[region > p].size == 0:
                r.append(x)
                c.append(y)

    return H, r, c

def plot_rotation(imgs: list, lbls: list):
    for i, img in enumerate(imgs):
        _, r, c = detect_corners(img)
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.scatter(c,r, s=1, c='red')
        plt.suptitle(lbls[i])
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=(i == len(imgs) - 1))

def plot_full(img: np.ndarray):
    Ix, Iy = image_derivatives(img)
    _, r, c = detect_corners(img)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    ax1.imshow(Ix) ; ax1.set_title(r"$\mathcal{I}_x$")
    ax2.imshow(Iy) ; ax2.set_title(r"$\mathcal{I}_y$")
    ax3.imshow(img, cmap='gray') ; ax3.scatter(c,r, s=1, c='red') ; ax3.set_title("Detected corners")
    [ax.axis('off') for ax in (ax1,ax2,ax3)]

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_src = None

    while img_src is None:
        img_sel = input("Select image:\n1) Toy\n2) Doll\n[1/2]: ")
        if img_sel == '1':
            img_src = "./images/toy/0001.jpg"
        elif img_sel == '2':
            img_src = "./images/doll/0200.jpg"
    
    img = plt.imread(img_src)
    
    if input("Apply rotation? [y/N] ") == 'y':
        img45 = rotate(img, 45)
        img90 = rotate(img, 90)
        plot_rotation([img, img45, img90], [r"$%d\degree$" % i for i in (0, 45, 90)])
    else:
        plot_full(img)
