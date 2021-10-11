from RANSAC import ransac
from utils import create_affine_matrix, crop_nonzero, imread_rgb, imshow, nn_interp

import matplotlib.pyplot as plt
import numpy as np


def stitching(left_img: np.ndarray, right_img: np.ndarray,  A: np.ndarray) -> np.ndarray:
    d = 3 if left_img.ndim == 3 else 1

    # Make sure that both images will fit in the output.
    # We will crop the final result later.
    l_h, l_w = left_img.shape[:2]
    r_h, r_w = right_img.shape[:2]
    h, w = max(l_h, r_h), max(l_w, r_w)
    output = np.full((2*h + 2*w, 2*w + 2*h, d), -1, dtype=int)
    output[l_h:2*l_h, l_w:2*l_w, :] = left_img

    for (y1, x1) in np.ndindex(right_img.shape[:2]):
        xy1 = np.array([x1, y1, 1]).T[:, np.newaxis]
        xy2 = A @ xy1

        xy2 = xy2.flatten()
        (x2, y2) = round(xy2[0]), round(xy2[1])
        output[h + y2, w + x2, :] = right_img[y1, x1, :]

    mask = []
    for (y,x) in ((0,0), (0,r_w-1), (r_h-1,0)):
        xy1 = np.array([x, y, 1]).T[:, np.newaxis]
        xy2 = (A @ xy1).flatten()

        (x2, y2) = round(xy2[0]), round(xy2[1])
        mask.append((w+x2,h+y2))
    
    for c in range(d):
        output[..., c]= nn_interp(output[..., c], mask)

    return crop_nonzero(output)


if __name__ == "__main__":
    left_img = imread_rgb('left.jpg')
    right_img = imread_rgb('right.jpg')

    np.random.seed(42)
    m, t, most_in = ransac(right_img, left_img)
    print(f"Detected {most_in} inliers")

    A = create_affine_matrix(m, t)
    stitch = stitching(left_img, right_img, A)

    imshow(stitch)

    imshow(left_img, right_img, stitch, \
         ax_titles=('Left Image', 'Right Image', 'Stitched Result')
        )
