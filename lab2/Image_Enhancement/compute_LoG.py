import numpy as np
from scipy.signal import convolve2d
from gauss2D import gauss2D

def LoG(s, x, y): 
    return (-1 / (np.pi * s ** 4) * np.exp(-1 / (2 * s ** 2) * (x ** 2 + y ** 2)) * (1 - (x ** 2 + y ** 2) / (2 * s ** 2)))


def compute_LoG(image, LOG_type):
    KERNEL_SIZE = 5
    SIGMA = 0.5
    search = False
    conv2D = lambda x, y, *args, **kwargs: convolve2d(x, y, mode='same',
                                                      boundary='symm', *args,
                                                      **kwargs)
    if LOG_type == 1:
        G_smoothing_kernel = gauss2D(*(2 * [SIGMA]), KERNEL_SIZE)
        gaussian = conv2D(image, G_smoothing_kernel)
        # Discrete laplacian with a 5 point stencil
        stencil5 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        #stencil9 = np.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]])
        imOut = conv2D(gaussian, stencil5)
    elif LOG_type == 2:
        start = int((KERNEL_SIZE - 1) / 2)
        if search:
            sum_min = float('inf')
            for i in range(100000):
                x = np.arange(-start, start + 1) * 0.0001 * i * SIGMA
                y = x.copy()
                kernel = LoG(SIGMA, x[:, np.newaxis], y[np.newaxis, :])
                if np.abs(np.sum(kernel)) < sum_min:
                    sum_min = np.abs(np.sum(kernel))
                    param = i
            x = np.arange(-start, start + 1) * 0.0001 * param * SIGMA
        else:
            x = np.arange(-start, start + 1) * 0.7127
        y = x.copy()
        kernel = LoG(SIGMA, x[:, np.newaxis], y[np.newaxis, :])
        imOut = conv2D(image, kernel)
    elif LOG_type == 3:
        ratio = 1.6
        s1 = SIGMA
        s2 = s1 * ratio
        image1 = conv2D(image, gauss2D(*(2 * [s1]), KERNEL_SIZE))
        image2 = conv2D(image, gauss2D(*(2 * [s2]), KERNEL_SIZE))
        DoG = image1 - image2
        imOut = DoG
    return imOut

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image = plt.imread('images/image2.jpg')
    fig, axs = plt.subplots(1, 4, figsize=(15, 15))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    for i in range(1, 4):
        axs[i].imshow(compute_LoG(image, i), cmap='gray')
        axs[i].set_title(f'Method {i}')
    plt.show()
