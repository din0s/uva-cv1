from scipy.signal import convolve2d
import numpy as np


def compute_gradient(image):
    kernel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=np.float64)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float64)
    Gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, kernel_y, mode='same', boundary='symm')
    im_magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    im_direction = np.arctan2(Gy, Gx)
    return Gx, Gy, im_magnitude, im_direction


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2

    image = plt.imread('images/image2.jpg')
    G_x, G_y, mag, dire = compute_gradient(image)
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))

    axs[0][0].imshow(image, cmap='gray')
    axs[0][0].set_title('Initial Image', fontsize=20)

    axs[0][1].imshow(G_x, cmap='gray')
    axs[0][1].set_title('Gradient x', fontsize=20)

    axs[0][2].imshow(G_y, cmap='gray')
    axs[0][2].set_title('Gradient y', fontsize=20)
    
    norm_dire = cv2.normalize(dire, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    axs[1][0].imshow(mag, cmap='gray')
    axs[1][0].set_title('Gradient Magnitude', fontsize=20)

    axs[1][1].imshow(norm_dire, cmap='gray')
    axs[1][1].set_title('Gradient Direction', fontsize=20)

    mag_dire = cv2.normalize(mag, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = np.multiply(mag_dire, np.sin(norm_dire))
    G = np.multiply(mag_dire, np.cos(norm_dire))
    B = np.zeros_like(R)
    color_dire = np.dstack((R, G, B))
    color_dire = cv2.normalize(color_dire, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    axs[1][2].imshow(color_dire)
    axs[1][2].set_title('Gradient Magnitude and Direction', fontsize=20)


    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()