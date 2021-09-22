import numpy as np

def gauss1D(sigma , kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    
    x = np.linspace(int(-kernel_size / 2), int(kernel_size / 2), kernel_size)
    G = np.exp((-1) * x ** 2 / (2 * sigma ** 2))
    G *= 1 / (sigma * np.sqrt(2 * np.pi))
    G /= np.sum(G)

    return G
