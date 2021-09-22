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


def median_filtering(image, n_channels, kernel_size):  # TO BE VECTORIZED!
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


# image = cv2.imread('images/image1_saltpepper.jpg')
image = cv2.imread('images/kobi.png')
image = image[:, :, ::-1] # for RGB
image_shape = image.shape
kwargs = {"kernel_size": 3, "sigma_x": 5, "sigma_y": 5}
kwargs = {"ksize": 9}
filtering_type = 'median'
denoised_image = denoise(image, filtering_type, **kwargs)

assert (denoised_image.shape == image_shape), "Different Shapes!"
assert (~np.array_equal(denoised_image, image)), "Image not filtered!"
assert (image_shape == image.shape), "Image shape changed!"

fig, axs = plt.subplots(1, 2, figsize=(15, 15))
axs[0].imshow(image)
axs[0].set_title('Original Image')

axs[1].imshow(denoised_image)
axs[1].set_title('Denoised Image ('+filtering_type+')')

plt.show()
