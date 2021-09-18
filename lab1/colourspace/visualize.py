import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize(input_image, cmap=None):
    if cmap == 'hsv':
        visualize_hsv(input_image)
    elif cmap == 'ycbcr':
        visualize_ycbcr(input_image)
    elif cmap == 'rgb':
        visualize_rgb(input_image)
    elif cmap == 'opponent':  # to be implemented!!
        visualize_opponent(input_image)
    elif cmap == 'gray':
        visualize_gray(input_image)
    else:
        print('Error: Unknown colorspace type [%s]...' % cmap)


def visualize_hsv(input_image):
    MAX_H = 179
    MAX_S = 255
    MAX_V = 255
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(input_image)

    # H, 1, 1
    h_image = np.ones_like(input_image)
    h_image[:, :, 0] = input_image[:, :, 0]
    h_image[:, :, 1] = MAX_S
    h_image[:, :, 2] = MAX_V
    axs[1].imshow(hsv2rgb(h_image))
    axs[1].set_title('Hue')

    # 1, S, 1
    s_image = np.empty_like(input_image)
    s_image[:, :, 0] = MAX_H
    s_image[:, :, 1] = input_image[:, :, 1]
    s_image[:, :, 2] = MAX_V
    axs[2].imshow(hsv2rgb(s_image))
    axs[2].set_title('Saturation')

    # 1, 0, V
    v_image = np.empty_like(input_image)
    v_image[:, :, 0] = MAX_H
    v_image[:, :, 1] = 0
    v_image[:, :, 2] = input_image[:, :, 2]
    axs[3].imshow(hsv2rgb(v_image))
    axs[3].set_title('Value')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def visualize_ycbcr(input_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(input_image)
    axs[0].set_title('Initial Image')

    y_image = np.ones_like(input_image)
    y_image[:, :, 0] = input_image[:, :, 0]
    y_image[:, :, 1] = 128
    y_image[:, :, 2] = 128
    axs[1].imshow(ycrcb2rgb(y_image))
    axs[1].set_title('Luma')

    cb_image = np.empty_like(input_image)
    cb_image[:, :, 0] = 128
    cb_image[:, :, 1] = input_image[:, :, 1]
    cb_image[:, :, 2] = 128
    axs[2].imshow(ycrcb2rgb(cb_image))
    axs[2].set_title('Blue Difference')

    cr_image = np.empty_like(input_image)
    cr_image[:, :, 0] = 128
    cr_image[:, :, 1] = 128
    cr_image[:, :, 2] = input_image[:, :, 2]
    axs[3].imshow(ycrcb2rgb(cr_image))
    axs[3].set_title('Red Difference')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def visualize_opponent(input_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    normalized_input_image = input_image.copy()
    normalized_input_image[:, :, 0] = min_max_normalization(normalized_input_image[:, :, 0])
    normalized_input_image[:, :, 1] = min_max_normalization(normalized_input_image[:, :, 1])
    normalized_input_image[:, :, 2] = min_max_normalization(normalized_input_image[:, :, 2])
    axs[0].imshow(normalized_input_image)

    rg_image = np.ones_like(input_image)
    r, g, b = redgreen2rgb(input_image[:, :, 0])
    rg_image[:, :, 0] = r
    rg_image[:, :, 1] = g
    rg_image[:, :, 2] = b

    axs[1].imshow(rg_image)
    axs[1].set_title('Red-Green')

    yb_image = np.empty_like(input_image)
    r, g, b = yellowblue2rgb(input_image[:, :, 1])
    yb_image[:, :, 0] = r
    yb_image[:, :, 1] = g
    yb_image[:, :, 2] = b

    axs[2].imshow(yb_image)
    axs[2].set_title('Yellow-Blue')

    l_image = np.empty_like(input_image)
    l_image[:, :, 0] = min_max_normalization(input_image[:, :, 2])
    l_image[:, :, 1] = min_max_normalization(input_image[:, :, 2])
    l_image[:, :, 2] = min_max_normalization(input_image[:, :, 2])
    axs[3].imshow(l_image)
    axs[3].set_title('Luminance')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def visualize_rgb(input_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(input_image)
    axs[0].set_title('Normalized Image')

    axs[1].imshow(input_image[:, :, 0], cmap='Reds')
    axs[1].set_title('Reds')

    axs[2].imshow(input_image[:, :, 1], cmap='Greens')
    axs[2].set_title('Greens')

    axs[3].imshow(input_image[:, :, 2], cmap='Blues')
    axs[3].set_title('Blues')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def visualize_gray(input_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(input_image[:, :, 0], cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Ligtness Method')

    axs[1].imshow(input_image[:, :, 1], cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('Average Method')

    axs[2].imshow(input_image[:, :, 2], cmap='gray', vmin=0, vmax=255)
    axs[2].set_title('Luminosity Method')

    axs[3].imshow(input_image[:, :, 3], cmap='gray', vmin=0, vmax=255)
    axs[3].set_title('Built-in Opencv Function')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


# utils
def ycbcr2ycrcb(input_image):
    cr = input_image[:, :, 2].copy()
    input_image[:, :, 2] = input_image[:, :, 1]
    input_image[:, :, 1] = cr

    return input_image


def hsv2rgb(hsv_image):
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) / 255.0


def ycrcb2rgb(ycbcr_image):
    return cv2.cvtColor(ycbcr2ycrcb(ycbcr_image), cv2.COLOR_YCrCb2RGB) / 255.0


def min_max_normalization(input_image):
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
    return input_image


def redgreen2rgb(input_image):
    red = input_image * np.sqrt(2)
    green = -1*red
    blue = 0.

    # red[red < 0.] = 0.
    # green[green < 0.] = 0.

    red = red/255.
    blue = blue/255.

    red = min_max_normalization(red)
    green = min_max_normalization(green)
    return red, green, blue


def yellowblue2rgb(input_image):
    yellow = input_image * (np.sqrt(6)/np.sqrt(2))
    blue = -1*yellow

    yellow = yellow / 255.
    blue = blue / 255.

    yellow = min_max_normalization(yellow)
    blue = min_max_normalization(blue)

    # yellow[yellow < 0.] = 0.
    # blue[blue < 0.] = 0.



    red = yellow
    green = yellow

    return red, green, blue





