"""
    Lucas Kanade Algorithm
    Python script for CV1, lab assignment 3
    Written by Rick Groenendijk
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from harris_corner_detector import HarrisCornerDetector
from lucas_kanade import LucasOpticalFlow

# Globals
POSSIBLE_EXTENSIONS = ('jpg', 'jpeg', 'ppm', 'pgm')
UPDATE_SCALING = 3


class Tracker(object):

    """
        Combines the Harris Corner Detector and the Lucas Kanade algorithm for
        optical flow into one tracker class. This class accepts a directory with sorted
        files and returns a sequence of frames in which the optical flow is computed
        at specific corner points.
    """

    def __init__(self, harris_threshold: float, gaussian_kernel_size: int = 3,
                 gaussian_kernel_sigma: float = 1.0, harris_patch_size: int = 5, window_size: int = 15) -> None:
        """ Initializes the tracker, using all parameters that can be passed along to the
            corner detector and optical flow class.
        :param harris_threshold: The threshold to decide whether element p of H is a corner.
        :param gaussian_kernel_size: The size of the kernel G in [size, size].
        :param gaussian_kernel_sigma: The sigma for the kernel G.
        :param harris_patch_size: The patch size [size, size] in which to look for corners.
        :param window_size: The window size of a single patch. 15 is the default, like in the assignment.
        """
        self.detector = HarrisCornerDetector(harris_threshold, gaussian_kernel_size=gaussian_kernel_size,
                                             gaussian_kernel_sigma=gaussian_kernel_sigma, harris_patch_size=harris_patch_size)
        self.flow = LucasOpticalFlow(window_size=window_size)
        self.window_size = window_size

    def track(self, dir_name: str, save_movie: bool = False, make_plot: bool = False) -> None:
        """ Given a directory, tracks the optical flow at corner points in frames.
            We detect corners in the first frame of the sequence and follow these corners
            throughout the sequence using the estimates of optical flow.
        :param dir_name: The relative directory string name.
        :param save_movie: Boolean to see if you want to save the frames to a movie.
        :param make_plot: Boolean to see if we should use matplotlib to make a plot (for debugging mostly).
        """
        # First, do a basic check on the directory.
        assert os.path.exists(dir_name) and os.path.isdir(dir_name), "{} has not been found".format(dir_name)

        # Get a list of files in the dir, make sure they are sorted.
        files_in_dir = os.listdir(dir_name)
        files_in_dir = sorted([os.path.join(dir_name, file) for file in files_in_dir
                               if file.split('.')[-1] in POSSIBLE_EXTENSIONS])

        # Detect the corners on the first frame.
        first_image = self._open_image(files_in_dir[0])
        _, r, c = self.detector.detect_corners(first_image)
        corners = np.vstack((r, c))
        # Filter the corners, such that corners at the edges of the images are not considered.
        corners = self._filter_corners_at_edge_image(first_image, corners)

        # Make an array for saving the movie frames.
        movie_frames = []
        # Mainly for debugging purposes, check how the frames look.
        if save_movie or make_plot:
            fig = plt.figure()
            plt.imshow(first_image)
            plt.scatter(corners[1],corners[0],c='red',marker='.')
            plt.draw()
            plt.axis("off")
            # For some reason pyplot needs a small pause to update the canvas.
            plt.pause(0.0001)
            img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_plot = np.reshape(img_plot, fig.canvas.get_width_height()[::-1] + (3,))
            movie_frames.append(img_plot)
            if make_plot:
                plt.show(block=False)

        del first_image
        # Loop over the images in the directory.
        for index in range(1, len(files_in_dir)):
            # Open the images at t - 1, t.
            previous_image = self._open_image(files_in_dir[index - 1])
            current_image = self._open_image(files_in_dir[index])

            # Get the optical flow at r, c. Resulting vector is [v_x, v_y].
            velocities = self.flow.determine_optical_flow(previous_image, current_image, points=corners)

            # Update the corners by the velocities. Since the image is in discrete space I choose
            # to update the corners only by integer values, so I round the velocities to the
            # nearest integer. Since we determine optical flow at the start of the sequence only,
            # there is bound to be a drift anyway.
            # Also, corners are [y x] where velocities are [v_x v_y], so add to the correct
            # rows.
            corners[1, :] += np.round(velocities[:, 0] * UPDATE_SCALING).astype(np.int)
            corners[0, :] += np.round(velocities[:, 1] * UPDATE_SCALING).astype(np.int)

            # Mainly for debugging purposes, check how the frames look.
            if save_movie or make_plot:
                # Clears the entire current figure with all its axes, but leaves the window.
                plt.clf()
                plt.imshow(current_image)
                plt.quiver(corners[1, :], corners[0, :], velocities[:, 0], velocities[:, 1],
                           angles='xy', scale_units='xy', scale=0.01)
                plt.draw()
                plt.axis("off")
                # For some reason pyplot needs a small pause to update the canvas.
                plt.pause(0.0001)
                img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_plot = np.reshape(img_plot, fig.canvas.get_width_height()[::-1]+(3,))
                movie_frames.append(img_plot)

            # Check if a corner has drifted to the edge of the image. If so, remove it.
            corners = self._filter_corners_at_edge_image(current_image, corners)

        if save_movie:
            movie_frames = np.stack(movie_frames)
            self._save_video("person_toy.avi", movie_frames, fps=3)

    def _filter_corners_at_edge_image(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """ We remove corners that are at the edges of the image, such that only full-sized
            patches are considered.
        :param image: The original image, to get the dimensions.
        :param corners: The corners that are detected by the Harris Corner detection algorithm.
        :return filtered_corners: The set of corners that is not close to the edge of the image.
        """
        # Filter the points that lie at the edges of the image, such that no window can bound them.
        if len(image.shape) == 3 and image.shape[-1] == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        from_edge = self.window_size // 2
        r, c = corners[0], corners[1]
        # The filter takes into account corners that lie at the edges of the image (left, right, top, bottom).
        filter = (from_edge < r) & (r < height - from_edge) & (from_edge < c) & (c < width - from_edge)
        # Apply the filter, to end up with a [2, N] array of corner points around which
        # patches should be centered.
        filtered_corners = corners[:, filter]
        return filtered_corners

    @staticmethod
    def _open_image(path):
        image = Image.open(path)
        return np.array(image)

    @staticmethod
    def _save_video(filename, array, fps=5):
        f, height, width, c = array.shape
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for i in range(f):
            out.write(array[i, :, :, ::-1])


if __name__ == '__main__':
    # Good Harris thresholds for the movies seem to be:
    # - person_toy: 0.001
    # - pingpong:   0.005
    tracker = Tracker(0.001)
    tracker.track('images/person_toy/', make_plot=True, save_movie=False)

    print('YOU ARE TERMINATED!')
