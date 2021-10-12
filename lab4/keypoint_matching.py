from utils import imread_gray, imshow

import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_kps_descs(img: np.ndarray, plot: bool = False) -> tuple:
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img, None)

    if plot:
        x = [kp.pt[0] for kp in kps]
        y = [kp.pt[1] for kp in kps]
        plt.scatter(x, y, c='red', s=1)
        imshow(img)

    return kps, descs


# default kp_thresh according to SIFT paper (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
def match_keypoints(*imgs: np.ndarray, kp_thresh: float = 0.7, plot: bool = False) -> tuple:
    keypoints = []
    descriptors = []
    for img in imgs:
        kps, descs = detect_kps_descs(img, plot)
        keypoints.append(np.array(kps))
        descriptors.append(descs)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(*descriptors, k=2)

    valid = []
    for (p1, p2) in matches:
        if p1.distance < kp_thresh * p2.distance:
            valid.append(p1)

    return keypoints, np.array(valid)


if __name__ == "__main__":
    img1 = imread_gray('boat1.pgm')
    img2 = imread_gray('boat2.pgm')

    kps, val = match_keypoints(img1, img2)

    plt.title("All valid matches")
    match = cv2.drawMatches(img1, kps[0], img2, kps[1], val, outImg=None, flags=2)
    imshow(match)

    np.random.seed(42)
    ind = np.random.choice(len(val), size=10, replace=False)

    plt.title("10 random samples")
    match_10 = cv2.drawMatches(img1, kps[0], img2, kps[1], val[ind], outImg=None, flags=2)
    imshow(match_10)
