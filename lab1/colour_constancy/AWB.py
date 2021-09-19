import numpy as np
import cv2
import sys
sys.path.append("../")
from colourspace.getColourChannels import getColourChannels
import matplotlib.pyplot as plt


def Grey_World1(input_image):
    input_image = input_image.astype(np.float32)
    R, G, B = getColourChannels(input_image)

    new_image = np.zeros_like(input_image)
    R_avg = np.mean(R)
    G_avg = np.mean(G)
    B_avg = np.mean(B)

    new_image[:, :, 0] = R / R_avg
    new_image[:, :, 1] = G / G_avg
    new_image[:, :, 2] = B / B_avg

    new_image = cv2.normalize(new_image, new_image, 0., 1., cv2.NORM_MINMAX).astype(float)

    return new_image


def Grey_World2(input_image):
    """https://www.researchgate.net/publication/269378413_Automatic_White_Balance_Based_on_Gray_World_Method_and_Retinex"""
    R, G, B = getColourChannels(input_image)

    new_image = np.zeros_like(input_image)

    a1 = min(np.mean(G) / np.mean(R), 1)
    b1 = min(np.mean(G) / np.mean(B), 1)

    new_image[:, :, 0] = a1 * R
    new_image[:, :, 1] = G
    new_image[:, :, 2] = b1 * B

    return new_image


def visualize(initial_image, grey_wolrd_image):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image', fontsize=18)

    axs[1].imshow(grey_wolrd_image)
    axs[1].set_title('Grey World Image', fontsize=18)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def visualize_all_algorithms(initial_image):
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))

    gw1 = Grey_World1(initial_image)
    gw2 = Grey_World2(initial_image)


    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image', fontsize=18)

    axs[1].imshow(gw1)
    axs[1].set_title('Grey World', fontsize=18)

    axs[2].imshow(gw2)
    axs[2].set_title('Grey World (2nd implementation)', fontsize=18)

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("awb.jpg")

    #convert from BGR to RGB
    image = image[:, :, ::-1]

    grey_world_image = Grey_World1(image)
    visualize(image, grey_world_image)

    # visualize_all_algorithms(image)




