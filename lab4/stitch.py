from RANSAC import ransac
from utils import affine_wrap, create_affine_matrix, imread_gray, imshow

import cv2
import numpy as np

if __name__ == "__main__":
    img2 = imread_gray('left.jpg')
    img1 = imread_gray('right.jpg')

    np.random.seed(42)
    m, t, most_in = ransac(img1, img2)
    print(most_in)
    A = create_affine_matrix(m, t)

    img2_aff = affine_wrap(img1, A)

    imshow(img1, img2_aff, img2)
