import numpy as np
import cv2
from colourspace.getColourChannels import getColourChannels
import matplotlib.pyplot as plt


def Grey_World(input_image):
    R, G, B = getColourChannels(input_image)

    new_image = np.zeros_like(input_image)
    R_avg = np.mean(R)
    G_avg = np.mean(G)
    B_avg = np.mean(B)
    a1 = G_avg / R_avg
    b1 = G_avg / B_avg
    new_image[:, :, 0] = a1 * R
    new_image[:, :, 1] = G
    new_image[:, :, 2] = b1 * B
    if np.allclose(new_image, input_image):
        print('Same pic')
    return new_image


def visualize(initial_image, grey_wolrd_image):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image')

    axs[1].imshow(grey_wolrd_image)
    axs[1].set_title('Gray World Image')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("colour_constancy/awb.jpg")

    #convert from BGR to RGB
    image = image[:, :, ::-1]

    grey_world_image = Grey_World(image)

    visualize(image, grey_world_image)

