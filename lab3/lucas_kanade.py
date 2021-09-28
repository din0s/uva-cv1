"""
    Lucas Kanade Algorithm
    Python script for CV1, lab assignment 3
    Written by Rick Groenendijk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from copy import deepcopy


class LucasOpticalFlow(object):

    """
        Implements the Lucas-Kanade Algorithm to determine optical flow.
    """

    def __init__(self, window_size: int = 15) -> None:
        """ Initializes the Lucas-Kanade method for optical flow.
        :param window_size: The window size of a single patch. 15 is the default, like in the assignment.
        """
        self.window_size = window_size

    def determine_optical_flow(self, I_0, I_1, make_plot: bool = False, points: np.ndarray = None) -> np.ndarray:
        """ Determines optical flow between a pair of images.
        :param I_0: Image at t=0 (or t - 1), that is the image at the previous time.
        :param I_1: Image at t=1 (or t), that is the image at the current time.
        :param make_plot: Plots the image and the optical flow vectors if True.
        :param points: The points at which to compute optical flow, if left as None, computes flow
                       uniformly across the image.
        :return v: The optical flow vectors [v_x, v_y]
        """

        # Convert the image from uint8 to float32, if necessary.
        if I_1.dtype == np.uint8:
            I_0 = I_0.astype(np.float32) / 255
            I_1 = I_1.astype(np.float32) / 255

        # Ensure image format is H x W.
        original_image = deepcopy(I_1)
        if len(I_1.shape) == 3 and I_1.shape[-1] == 3:
            I_0 = np.mean(I_0, axis=2)
            I_1 = np.mean(I_1, axis=2)
        elif len(I_0.shape) == 3 and I_0.shape[-1] == 1:
            I_0 = I_0[0]
            I_1 = I_1[0]

        # Compute matrices A, b.
        A, b = self._compute_system_variables(I_0, I_1, points=points)

        # Solve the systems.
        v = np.stack([self._solve_system(A[:, :, i], b[:, i]) for i in range(A.shape[-1])])

        # Optionally make a plot for the report.
        if make_plot:
            self._make_image_plot(original_image, v)
        return v

    def _compute_system_variables(self, I_0: np.ndarray, I_1: np.ndarray, points: np.ndarray = None) -> tuple:
        """ Computes parameters A, b of the system.
        :param I_0: Image at t=0 (or t - 1), that is the image at the previous time.
        :param I_1: Image at t=1 (or t), that is the image at the current time.
        :param points: The points at which to compute optical flow, if left as None, computes flow
                       uniformly across the image.
        :return: Matrices A, b for each of the regions.
        """
        # Obtain image x,y-derivatives.
        first_order_gaussian = np.array([[-1, 0, 1]])
        I_x = convolve2d(I_1, first_order_gaussian, mode='same')
        I_y = convolve2d(I_1, first_order_gaussian.T, mode='same')

        # Obtain time derivative as the subtraction between frames.
        I_t = I_0 - I_1

        # Obtain the pixel indices that index the pixels assigned to each non-overlapping patch.
        if not type(points) == np.ndarray and not points:
            pixel_indices_per_patch, num_patches = self._get_pixel_indices_uniform(I_1)
        else:
            pixel_indices_per_patch, num_patches = self._get_pixel_indices_at_points(I_1, points)

        # Apply these to the image derivatives.
        I_x_reshaped = I_x.flatten()[pixel_indices_per_patch]
        I_y_reshaped = I_y.flatten()[pixel_indices_per_patch]
        I_t_reshaped = I_t.flatten()[pixel_indices_per_patch]

        # Obtain A, b.
        A = np.moveaxis(
            np.dstack((I_x_reshaped.reshape(num_patches, self.window_size ** 2),
                       I_y_reshaped.reshape(num_patches, self.window_size ** 2))), 0, 2
        )
        b = np.moveaxis(I_t_reshaped.reshape((num_patches, self.window_size ** 2)), 0, 1) * -1.0
        return A, b

    def _get_pixel_indices_uniform(self, image: np.ndarray) -> tuple:
        """ Obtain the indices for each pixel belonging to a non-overlapping patch.
        :param image: The input image
        :return pixel_indices: a pixel-per-patch array with dimensions [window_size, window_size, amount_of_patches]
        :return num_patches: The total number of non-overlapping patches in the image.
        """

        # Splice the arrays, to obtain a [window_size, window_size, height // window_size * width // window_size]
        # array for each of the derivatives.
        height, width = image.shape
        num_row_patches, num_column_patches = height // self.window_size, width // self.window_size
        patch_indices = [(i, j) for i in range(num_row_patches) for j in range(num_column_patches)]

        # Get the general pixel indices for each patch.
        indices = np.linspace(0, height * width - 1, height * width).reshape((height, width)).astype(np.int)
        pixel_indices = [
            indices[i * self.window_size:(i + 1) * self.window_size, j * self.window_size:(j + 1) * self.window_size]
            for (i, j) in patch_indices]
        pixel_indices = np.stack(pixel_indices)
        return pixel_indices, num_row_patches * num_column_patches

    def _get_pixel_indices_at_points(self, image: np.ndarray, points: np.ndarray) -> tuple:
        """ Obtain the indices for each pixel belonging to a non-overlapping patch.
        :param image: The input image
        :param points: array with the indices of the points to center the window around.
        :return pixel_indices: a pixel-per-patch array with dimensions [window_size, window_size, amount_of_patches]
        :return num_patches: The total number of non-overlapping patches in the image.
        """
        _, width = image.shape
        # Now make a [N, window_size, window_size] array with the image pixel indices.
        pixel_indices = np.zeros((points.shape[1], self.window_size, self.window_size)).astype(np.int)
        for i in range(points.shape[1]):
            # Get the x,y pixel indices at the left upper corner of the patch.
            y, x = points[0, i] - self.window_size // 2, points[1, i] - self.window_size // 2
            # Make a patch of [window_size, window_size]
            y_indices = np.repeat(np.linspace(
                y, y + self.window_size - 1, self.window_size), self.window_size
            ).reshape((self.window_size, self.window_size))
            x_indices = np.repeat(np.linspace(
                x, x + self.window_size - 1, self.window_size), self.window_size
            ).reshape((self.window_size, self.window_size)).T
            # Now the y indices should be multiplied by the amount of columns, then add.
            patch = (y_indices * width + x_indices).astype(np.int)
            pixel_indices[i] = patch
        return pixel_indices, points.shape[1]

    def _make_image_plot(self, image: np.ndarray, v: np.ndarray) -> None:
        """ Makes a plot, according to exercise 2.1 in the assignment.
        :param image: The original image.
        :param v: The velocities matrix
        """
        plt.figure()
        plt.imshow(image)

        # Get locations of the vectors.
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        num_row_patches, num_column_patches = height // self.window_size, width // self.window_size
        patch_indices = np.stack(
            [(i, j) for i in range(num_row_patches) for j in range(num_column_patches)]
        )
        r = patch_indices[:, 0] * self.window_size + self.window_size // 2
        c = patch_indices[:, 1] * self.window_size + self.window_size // 2

        # Get the velocities in x,y and then plot using a quiver plot.
        v_x = v[:, 0]
        v_y = v[:, 1]
        plt.quiver(c, r, v_x, v_y, angles='xy', scale_units='xy', scale=0.1)
        plt.show()
        return

    @staticmethod
    def _solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(np.linalg.pinv(A), b)


if __name__ == '__main__':
    # Open an image.
    from PIL import Image
    image_object0 = Image.open('images/sphere1.ppm')
    image_numpy0 = np.array(image_object0)
    image_object1 = Image.open('images/sphere2.ppm')
    image_numpy1 = np.array(image_object1)

    # Init the Lucas Kanade method, compute optical flow, and show.
    optical = LucasOpticalFlow(15)
    optical.determine_optical_flow(image_numpy0, image_numpy1, make_plot=True)

    print('YOU ARE TERMINATED!')
