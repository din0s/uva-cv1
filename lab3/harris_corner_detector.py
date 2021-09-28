"""
    Harris Corner Detector
    Python script for CV1, lab assignment 3
    Written by Rick Groenendijk
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter
from copy import deepcopy


class HarrisCornerDetector(object):

    """
        Implements the Harris corner detector. Optionally plots results.
    """

    def __init__(self, harris_threshold: float, gaussian_kernel_size: int = 3,
                 gaussian_kernel_sigma: float = 1.0, harris_patch_size: int = 5) -> None:
        """ Initializes the class, and sets a number of hyper parameters for the algorithm.
        :param harris_threshold: The threshold to decide whether element p of H is a corner.
        :param gaussian_kernel_size: The size of the kernel G in [size, size].
        :param gaussian_kernel_sigma: The sigma for the kernel G.
        :param harris_patch_size: The patch size [size, size] in which to look for corners.
        """
        # Harris parameters.
        self.harris_threshold = harris_threshold
        self.harris_patch_size = harris_patch_size
        # Kernel parameters.
        self.kernel_size = gaussian_kernel_size
        self.kernel_sigma = gaussian_kernel_sigma

    def detect_corners(self, image: np.ndarray, make_plot: bool = False) -> tuple:
        """ Detects corners in the input image according to Harris.
        :param image: H x W x 3 or H x W image as a numpy array.
        :param make_plot: Plots the image derivatives and detected corner points.
        :return H: H X W matrix H
        :return r: row indices of the corners
        :return c: column indices of the corners
        """
        # Convert the image from uint8 to float32, if necessary.
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255

        # Ensure image format is H x W.
        original_image = deepcopy(image)
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = np.mean(image, axis=2)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image[0]

        # Obtain image derivatives.
        first_order_gaussian = np.array([[-1, 0, 1]])
        I_x = convolve2d(image, first_order_gaussian, mode='same')
        I_y = convolve2d(image, first_order_gaussian.T, mode='same')

        # Define the Gaussian filter G.
        G = self._obtain_gaussian_kernel()

        # Obtain matrices A, B, C in matrix Q
        A = convolve2d(I_x ** 2, G, mode='same')
        B = convolve2d((I_x * I_y), G, mode='same')
        C = convolve2d(I_y ** 2, G, mode='same')

        # Compute the H matrix using (A * C − B ** 2) − 0.04 * (A + C) ** 2 given by equation 12.
        H = A * C - B ** 2 - 0.04 * (A + C) ** 2

        # Finally, obtain the corners in a local neighbourhood.
        r, c = self._get_harris_points(H)

        # Optionally make a plot for the report.
        if make_plot:
            self._make_image_plot(original_image, I_x, I_y, r, c)
        return H, r, c

    def _get_harris_points(self, H: np.ndarray):
        """ Given matrix H, finds the corners that are local maxima in an image patch.
        :param H: The matrix H, obtained from detect_corners
        :return r, c: a tuple of row, column indices.
        """
        H_local_maxima = maximum_filter(H, size=self.harris_patch_size)
        H[H < H_local_maxima] = 0.0
        r, c = np.where(H > self.harris_threshold)
        return r, c

    @staticmethod
    def _make_image_plot(image: np.ndarray, I_x: np.ndarray, I_y: np.ndarray, r: np.ndarray, c: np.ndarray):
        """ Makes a plot, according to exercise 1.2 in the assignment.
        :param image: The original image.
        :param I_x: Image derivatives along x
        :param I_y: Image derivatives along y
        :param r: The row indices of corners points
        :param c: The column indices of corner points
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Plot the image derivatives
        axes[0].imshow(I_x)
        axes[0].set_title('I_x: Image Derivative along x')
        axes[1].imshow(I_y)
        axes[1].set_title('I_y: Image Derivative along y')

        # To make the plot a bit nicer, we cut off corners detected at the edges of the image.
        max_height, max_width, _ = image.shape
        r, c = zip(*[(i, j) for i, j in zip(list(r), list(c)) if (2 < i < max_height - 2 and 2 < j < max_width - 2)])

        # Plot the original image, showing the corners points.
        axes[2].imshow(image)
        axes[2].scatter(c, r, s=1, color='red')
        axes[2].set_title('Detected corner points')

        plt.tight_layout()
        plt.show()
        return

    def _obtain_gaussian_kernel(self):
        """ Make a Gaussian kernel of size [self.kernel_size, self.kernel_size] with
            a sigma of self.kernel_sigma.
            Obtained from: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        :return kernel: the Gaussian kernel.
        """
        interval = (2 * self.kernel_sigma + 1.) / self.kernel_size
        x = np.linspace(-self.kernel_sigma - interval / 2., self.kernel_sigma + interval / 2., self.kernel_size + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


if __name__ == '__main__':
    # Open an image.
    from PIL import Image
    image_object = Image.open('images/person_toy/00000001.jpg')
    image_numpy = np.array(image_object)

    # Init the Harris Corner detector, detect corners, and show.
    detector = HarrisCornerDetector(0.001)
    H, r, c = detector.detect_corners(image_numpy, make_plot=True)

    print('YOU ARE TERMINATED!')
