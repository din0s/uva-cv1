import numpy as np
import cv2
from getColourChannels import getColourChannels


def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    R, G, B = getColourChannels(input_image)

    # ligtness method
    maximum = np.maximum(np.maximum(R, G), B)
    minimum = np.minimum(np.minimum(R, G), B)
    ligtness = (maximum + minimum) / 2.

    # average method
    average = (R + G + B) / 3.

    # luminosity method
    luminosity = 0.21*R + 0.72*G + 0.07*B

    # built-in opencv function
    library = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    new_image = np.stack([ligtness, average, luminosity, library], axis=2)

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    new_image = np.zeros_like(input_image)
    R, G, B = getColourChannels(input_image)

    new_image[:, :, 0] = (R - G) / np.sqrt(2)
    new_image[:, :, 1] = (R + G - 2*B) / np.sqrt(6)
    new_image[:, :, 2] = (R + G + B) / np.sqrt(3)

    return new_image


def rgb2normedrgb(input_image):
    new_image = np.zeros_like(input_image)
    R, G, B = getColourChannels(input_image)

    normalizer = R + G + B
    # if normalizer=0 then R=G=B=0 (at a given point).
    # Therefore, setting these points of normalizer to 1 will not have an effect to the new image. (0/1 = 0)
    normalizer[normalizer == 0] = 1

    new_image[:, :, 0] = R / normalizer
    new_image[:, :, 1] = G / normalizer
    new_image[:, :, 2] = B / normalizer

    return new_image
