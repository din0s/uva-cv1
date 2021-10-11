import cv2
import matplotlib.pyplot as plt
import numpy as np


def imread_gray(file: str) -> np.ndarray:
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def imread_rgb(file: str) -> np.ndarray:
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imshow(*imgs: np.ndarray, ax_titles: list = ()):
    n, t = len(imgs), len(ax_titles)
    if n == 1:
        plt.imshow(imgs[0], cmap='gray')
        plt.axis('off')
    else:
        _, axs = plt.subplots(1, n, figsize=(5*n, 5))
        for i, img in enumerate(imgs):
            axs[i].imshow(img, cmap='gray')
            axs[i].set_axis_off()

            if i < t:
                axs[i].set_title(ax_titles[i])

    plt.tight_layout()
    plt.show()


def imshow_grid(*imgs: np.ndarray, shape: tuple = None, ax_titles: list = ()):
    n, t = len(imgs), len(ax_titles)
    if n == 1:
        plt.imshow(imgs[0], cmap='gray')
        plt.axis('off')
    else:
        if shape is None:
            shape = (1, n)
        size = tuple(np.dot(shape[::-1], 5))
        _, axs = plt.subplots(*shape, figsize=size)
        for i, row in enumerate(list(axs)):
            for j, ax in enumerate(row):
                index = i*len(axs) + j
                ax.imshow(imgs[index], cmap='gray')
                ax.set_axis_off()

                if index < t:
                    ax.set_title(ax_titles[index])

    plt.tight_layout()
    plt.show()


def create_affine_matrix(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    # rotation matrix
    R = np.zeros((3, 3))
    R[0:2, 0:2] = m
    R[2, 2] = 1

    # translation matrix
    T = np.eye(3)
    T[0:2, 2] = t.squeeze()
    # result is given by mat mult
    return T @ R


def nn_interp(image: np.ndarray, mask: list, invalid_val: int = -1) -> np.ndarray:
    if invalid_val != 0:
        image -= invalid_val

    A, B, D = mask
    # How to check if point is inside rectangle
    # https://math.stackexchange.com/a/190373
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    AD = np.array([D[0] - A[0], D[1] - A[1]])

    r, c = np.nonzero(image)
    for (yi, xi) in np.ndindex(image.shape[:2]):
        if image[yi, xi] == 0:
            AM = np.array([xi - A[0], yi - A[1]])
            ABM = AB @ AM
            ABB = AB @ AB
            ADM = AD @ AM
            ADD = AD @ AD

            if ABM > 0 and ABM < ABB and ADM > 0 and ADM < ADD:
                nn = ((r - yi)**2 + (c - xi)**2).argmin()
                image[yi, xi] = image[r[nn], c[nn]]

    if invalid_val != 0:
        image += invalid_val
    
    image[image == invalid_val] = 0
    return image

def affine_wrap(img: np.ndarray, A: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w, d = img.shape
    img_aff = np.full_like(img, -1, dtype=int)

    for (y1, x1) in np.ndindex(img.shape[:2]):
        xy1 = np.array([x1, y1, 1]).T[:, np.newaxis]
        xy2 = A @ xy1

        xy2 = xy2.flatten()
        (x2, y2) = round(xy2[0]), round(xy2[1])
        if x2 >= 0 and x2 < w and y2 >= 0 and y2 < h:
            img_aff[y2, x2, :] = img[y1, x1, :]
    
    mask = []
    for (y,x) in ((0,0), (0,w-1), (h-1,0)):
        xy1 = np.array([x, y, 1]).T[:, np.newaxis]
        xy2 = (A @ xy1).flatten()

        (x2, y2) = round(xy2[0]), round(xy2[1])
        mask.append((x2,y2))

    for c in range(d):
        img_aff[..., c] = nn_interp(img_aff[..., c], mask)

    return img_aff


def crop_nonzero(image: np.ndarray) -> np.ndarray:
    i = image
    if image.ndim == 3:
        i = np.mean(image, axis=2)
    y_nonzero, x_nonzero = np.nonzero(i)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
