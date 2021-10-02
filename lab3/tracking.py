from harris_corner_detector import detect_corners
from lucas_kanade import optical_flow

import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    vid_dir = None

    while vid_dir is None:
        vid_sel = input("Select video:\n1) Toy\n2) Doll\n[1/2]: ")
        if vid_sel == '1':
            vid_dir = "./images/toy"
        elif vid_sel == '2':
            vid_dir = "./images/doll"

    # read all the images from the given directory into a list
    imgs = [plt.imread(f"{vid_dir}/{i}") for i in os.listdir(vid_dir)]

    # detect points of interest (POI) from the first frame
    _, r, c = detect_corners(imgs[0])

    while len(imgs) >= 2:
        # stack POI in a [ point_row point_col ] fashion
        corners = np.vstack([r,c]).T
        # determine optical flow and retrieve new location of POI
        r, c = optical_flow(imgs[0], imgs[1], corners, blockPlot=False)
        # remove first image from the list and continue iterating
        imgs.pop(0)
