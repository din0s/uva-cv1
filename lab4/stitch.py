from RANSAC import ransac
from utils import create_affine_matrix, imread_gray, imshow, stitching

import numpy as np


if __name__ == "__main__":
    left_img = imread_gray('left.jpg')
    right_img = imread_gray('right.jpg')

    np.random.seed(42)
    m, t, most_in = ransac(right_img, left_img)
    print(most_in)
    A = create_affine_matrix(m, t)

    stitch = stitching(left_img, right_img, A)

    imshow(left_img, right_img, stitch)


