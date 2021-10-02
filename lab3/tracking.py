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
        elif vid_sel == '3':
            vid_dir = '/home/john/cv1_labs/lab3/images/pingpong'
        elif vid_sel == '4':
            vid_dir = '/home/john/cv1_labs/lab3/images/person_toy'

    # read all the images from the given directory into a list
    imgs = [plt.imread(f"{vid_dir}/{i}") for i in sorted(os.listdir(vid_dir))]

    # detect points of interest (POI) from the first frame
    _, r, c = detect_corners(imgs[0])
    
    remaindervxs, remaindervys, vxs, vys = np.array([]), np.array([]), np.array([]), np.array([])
    while len(imgs) >= 2:
        # stack POI in a [ point_row point_col ] fashion
        if vxs.any() and vys.any():
            if remaindervxs.any():
                remaindervxs = remaindervxs[inside]
            if remaindervys.any():
                remaindervys = remaindervys[inside]
            wholevxs = np.round(vxs, 0).astype(int)
            wholevys = np.round(vys, 0).astype(int)
            remaindervxs = remaindervxs + vxs - wholevxs.astype(np.float32) if remaindervxs.any() else vxs - wholevxs.astype(np.float32)
            remaindervys = remaindervys + vys - wholevys.astype(np.float32) if remaindervys.any() else vys - wholevxs.astype(np.float32)
            vxs += remaindervxs
            print(wholevxs.any())
            r += wholevys
            vys += remaindervys
            c += wholevxs
        corners = np.vstack([r,c]).T
        # determine optical flow and retrieve new location of POI
        r, c, vxs, vys, inside = optical_flow(imgs[0], imgs[1], corners, blockPlot=False)
        # remove first image from the list and continue iterating
        imgs.pop(0)
