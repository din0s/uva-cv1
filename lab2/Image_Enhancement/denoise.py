import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from gauss2D import gauss2D


def box_kernel(kernel_size):
    return np.ones((kernel_size, kernel_size)) / kernel_size**2


def filtering_image(image, kernel, n_channels):
    denoised_image = np.zeros_like(image)
    for channel in range(n_channels):
        denoised_image[:, :, channel] = signal.convolve2d(image[:, :, channel], kernel, mode='same')
    return denoised_image


def median_filtering(image, n_channels, kernel_size):
    pad_size = int(kernel_size/2)
    padded_image = np.pad(image, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], 'edge')
    denoised_image = np.zeros_like(image)

    for channel in range(n_channels):
        for i in range(int(kernel_size/2), image.shape[0]):
            for j in range(int(kernel_size/2), image.shape[1]):
                denoised_image[i, j, channel] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size, channel])
    return denoised_image


def denoise(image, kernel_type, **kwargs):
    n_channels = image.shape[-1] if len(image.shape) > 2 else 1
    image = image.reshape(image.shape[0], image.shape[1], n_channels)

    if kernel_type == 'box':
        # kernel = box_kernel(**kwargs)
        # denoised_image = filtering_image(image, kernel, n_channels)
        denoised_image = cv2.blur(image, **kwargs)
    elif kernel_type == 'median':
        # denoised_image = median_filtering(image, n_channels, **kwargs)
        denoised_image = cv2.medianBlur(image, **kwargs)
    elif kernel_type == 'gaussian':
        kernel = gauss2D(**kwargs)
        denoised_image = filtering_image(image, kernel, n_channels)
    else:
        print('Operation not implemented')

    return denoised_image
