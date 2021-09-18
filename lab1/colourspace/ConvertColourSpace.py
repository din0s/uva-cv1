import rgbConversions
from visualize import *
import cv2


def ycrcb2ycbcr(input_image):
    cb = input_image[:, :, 2].copy()
    input_image[:, :, 2] = input_image[:, :, 1]
    input_image[:, :, 1] = cb

    return input_image


def ConvertColourSpace(input_image, colourspace):
    '''
    Converts an RGB image into a specified color space, visualizes the
    color channels and returns the image in its new color space.

    Colorspace options:
      opponent
      rgb -> for normalized RGB
      hsv
      ycbcr
      gray

    '''

    if colourspace.lower() == 'opponent':
        input_image = input_image.astype(np.float32)
        new_image = rgbConversions.rgb2opponent(input_image)

    elif colourspace.lower() == 'rgb':
        input_image = input_image.astype(np.float32)
        new_image = rgbConversions.rgb2normedrgb(input_image)

    elif colourspace.lower() == 'hsv':
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

    elif colourspace.lower() == 'ycbcr':
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
        # convert from ycrcb to ycbcr
        new_image = ycrcb2ycbcr(new_image)

    elif colourspace.lower() == 'gray':
        input_image = input_image.astype(np.float32)
        new_image = rgbConversions.rgb2grays(input_image)

    else:
        print('Error: Unknown colorspace type [%s]...' % colourspace)
        new_image = input_image

    visualize(new_image, cmap=colourspace)

    return new_image


if __name__ == '__main__':
    img_path = 'orgb.png'
    I = cv2.imread(img_path)
    # Convert from BGR to RGB
    I = I[:, :, ::-1]

    out_img = ConvertColourSpace(I, 'opponent')
