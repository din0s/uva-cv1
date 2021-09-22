from scipy.signal import convolve2d
import numpy as np


def compute_gradient(image):
    kernel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=np.float64)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float64)
    Gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, kernel_y, mode='same', boundary='symm')
    im_magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    Gx[Gx == 0] = 1e-16
    im_direction = np.arctan(Gy / Gx)
    return Gx, Gy, im_magnitude, im_direction


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    image = plt.imread('images/image2.jpg')
    G_x, G_y, mag, dire = compute_gradient(image)
    fig, axs = plt.subplots(1, 5, figsize=(15, 15))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Initial Image')

    axs[1].imshow(G_x, cmap='gray')
    axs[1].set_title('Gradient x')

    axs[2].imshow(G_y, cmap='gray')
    axs[2].set_title('Gradient y')

    axs[3].imshow(mag, cmap='gray')
    axs[3].set_title('Gradient Magnitude')
    
    norm_dire = cv2.normalize(dire, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    axs[4].imshow(norm_dire, cmap='gray')
    axs[4].set_title('Gradient Direction')
    
    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
