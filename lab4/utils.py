import cv2
import matplotlib.pyplot as plt
import numpy as np

def imread_gray(file: str) -> np.ndarray:
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def imshow(*imgs: np.ndarray, ax_titles: list = (), cmap: str = 'gray'):
    n, t = len(imgs), len(ax_titles)
    if n == 1:
        plt.imshow(imgs[0], cmap)
    else:
        _, axs = plt.subplots(1, n, figsize=(5*n, 5))
        for i, img in enumerate(imgs):
            axs[i].imshow(img, cmap)

            if i < t:
                axs[i].set_title(ax_titles[i])

    plt.tight_layout()
    plt.show()
    plt.clf()


def create_affine_matrix(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    # rotation matrix
    R = np.zeros((3, 3))
    R[0:2, 0:2] = m
    R[2, 2] = 1

    # translation matrix
    T = np.eye(3)
    T[0:2, 2] = t.squeeze()

    # result is given by mat mult
    return R @ T

def affine_wrap(img: np.ndarray, A: np.ndarray) -> np.ndarray:
    h, w = img.shape
    img_aff = np.zeros_like(img)
    for (y1, x1), val in np.ndenumerate(img):
        xy1 = np.array([x1, y1,1]).T[:, np.newaxis]
        xy2 = A @ xy1

        xy2 = xy2.flatten()
        (x2, y2) = round(xy2[0]), round(xy2[1])
        if x2 >= 0 and x2 < w and y2 >= 0 and y2 < h:
            img_aff[y2, x2] = val

    return img_aff
