import numpy as np
import cv2
from colourspace.getColourChannels import getColourChannels
import matplotlib.pyplot as plt


def Grey_World2(input_image):
    R, G, B = getColourChannels(input_image)

    new_image = np.zeros_like(input_image)
    R_avg = np.mean(R)
    G_avg = np.mean(G)
    B_avg = np.mean(B)
    Gray_world = (R_avg + G_avg + B_avg) / 3
    normalize = np.vectorize(lambda x: 255 if x > 255 else int(x))
    new_image[:, :, 0] = normalize((Gray_world / R_avg) * R)
    new_image[:, :, 1] = normalize((Gray_world / G_avg) * G)
    new_image[:, :, 2] = normalize((Gray_world / B_avg) * B)
    return new_image


def Grey_World3(input_image):
    R, G, B = getColourChannels(input_image)

    new_image = np.zeros_like(input_image)

    a1 = min(np.mean(G) / np.mean(R), 1)
    b1 = min(np.mean(G) / np.mean(B), 1)

    new_image[:, :, 0] = a1 * R
    new_image[:, :, 1] = G
    new_image[:, :, 2] = b1 * B

    return new_image


def Grey_World(input_image):
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


def visualize(initial_image, grey_wolrd_image):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image', fontsize=20)

    axs[1].imshow(grey_wolrd_image)
    axs[1].set_title('Grey World Image', fontsize=20)

    #fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("awb.jpg")

    #convert from BGR to RGB
    image = image[:, :, ::-1]
    grey_world_image = Grey_World(image)
    visualize(image, grey_world_image)




