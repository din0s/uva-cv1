from keypoint_matching import match_keypoints
from utils import affine_wrap, create_affine_matrix, imread_gray, imshow

import cv2
import numpy as np

def ransac(*imgs: np.ndarray, N: int = 50, P: int = 10, radius: float = 10.0, plot: bool = False) -> tuple:
    kps, valid = match_keypoints(*imgs)

    most_inline = -1
    m_best, t_best = None, None

    for _ in range(N):
        ind = np.random.choice(len(valid), size=P, replace=False)
        sample = valid[ind]

        XY1, XY2 = [], []
        A, b = [], []
        for p in sample:
            kp1 = kps[0][p.queryIdx]
            kp2 = kps[1][p.trainIdx]
            x1, y1 = kp1.pt[0], kp1.pt[1]
            x2, y2 = kp2.pt[0], kp2.pt[1]

            XY1.append([x1, y1, 1])
            XY2.append([x2, y2, 1])

            A.append([[x1,y1,0,0,1,0], [0,0,x1,y1,0,1]])
            b.append([[x2, y2]])

        A = np.array(A).reshape(-1, 6)
        b = np.array(b).flatten().T
        x = np.linalg.pinv(A) @ b

        m = np.array([[x[0], x[1]], [x[2], x[3]]])
        t = np.array([[x[4], x[5]]]).T

        XY2_aff = XY1 @ create_affine_matrix(m, t)

        err = np.abs(XY2 - XY2_aff)
        err_bin = np.sqrt(err[0, :]**2 + err[1, :]**2) <= radius
        inline_count = err_bin.sum()

        if inline_count > most_inline:
            most_inline = inline_count
            m_best = m
            t_best = t

        if plot:
            comp = cv2.drawMatches(imgs[0], kps[0], imgs[1], kps[1], sample, outImg=None)
            imshow(comp)

    # most_inline needs to be at least 3 (3 transformation matches)!
    return m_best, t_best, most_inline

if __name__ == "__main__":
    img1 = imread_gray('boat1.pgm')
    img2 = imread_gray('boat2.pgm')

    np.random.seed(42)
    m, t, most_in = ransac(img1, img2)
    A = create_affine_matrix(m, t)

    img2_aff = affine_wrap(img1, A)
    img2_cv2 = cv2.warpAffine(img1, A[:-1, :], img2.shape[::-1])

    imshow(img1, img2_aff, img2_cv2, img2, \
         ax_titles=("Original image", "Our implementation", "OpenCV warpAffine", "Ground truth") \
        )
