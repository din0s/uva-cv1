import numpy as np
import matplotlib.pyplot as plt
from utils import image_derivatives

def blockify(image: np.ndarray, stride: int) -> np.ndarray:
    x, y = image.shape
    blocks = np.array([image[i:i + stride, j:j + stride] for j in range(0, y - y % stride, stride)
			   for i in range(0, x - x % stride,stride)],)
    blocks = blocks.reshape(blocks.shape[0], -1)
    return blocks


def lucas_kanade(image1: np.ndarray, image2: np.ndarray) -> None:
    rgb_image = image2
    STRIDE = 15
    image1 = np.mean(image1, axis=2).astype(np.float32) / 255
    image2 = np.mean(image2, axis=2).astype(np.float32) / 255

    G_x, G_y = image_derivatives(image1)
    blocks_G_x = blockify(G_x, STRIDE)
    blocks_G_y = blockify(G_y, STRIDE)
    As = [np.array([A, B]).T for A, B in zip(blocks_G_x, blocks_G_y)]
    Bs = blockify(image2 - image1, STRIDE)
    vs = [np.linalg.pinv(A) @ b for (A, b) in zip(As, Bs)]
    plt.imshow(image1, cmap='gray')
    centers = [[(x, y) for y in range(STRIDE // 2, image1.shape[0] - image1.shape[0] % STRIDE, STRIDE)] for x in range(STRIDE // 2, image1.shape[1] - image1.shape[1] % STRIDE, STRIDE)]
    centers = sum(centers, [])
    x, y = np.array([l[0] for l in centers]), np.array([l[1] for l in centers])
    plt.imshow(rgb_image)
    plt.quiver(x, y, np.array(vs)[:, 0], np.array(vs)[:, 1], angles='xy', scale_units='xy', scale=0.1)
    plt.show()

if __name__ == '__main__':
	image1 = plt.imread('images/Car1.jpg')
	image2 = plt.imread('images/Car2.jpg')
	lucas_kanade(image1, image2)