from harris_corner_detector import detect_corners
from lucas_kanade import optical_flow

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import mask_centers

def track_flow(imgs: list, vid_name: str, harris_thresh: float = 1e-5, save_vid: bool = False):
    h, w, *_ = imgs[0].shape

    # detect points of interest (POI) from the first frame
    _, r, c = detect_corners(imgs[0], threshold=harris_thresh)

    fig_data = []
    vxs_rem, vys_rem = None, None
    while len(imgs) >= 2:
        # stack POI in a [ point_row point_col ] fashion
        corners = np.vstack([r,c]).T

        if save_vid:
            # determine optical flow and retrieve new location of POI
            vxs, vys, fig = optical_flow(imgs[0], imgs[1], corners, video_mode=True)

            # encode canvas data and append to list to create a vid later
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig_data.append(data)
        else:
            # determine optical flow and retrieve new location of POI
            vxs, vys = optical_flow(imgs[0], imgs[1], corners, block_plot=False)

        if vxs_rem is None:
            vxs_rem = np.zeros_like(vxs)
            vys_rem = np.zeros_like(vys)

        # calculate whole 'integer' pixel movements
        vxs_whole = np.around(vxs + vxs_rem, decimals=0).astype(int)
        vys_whole = np.around(vys + vys_rem, decimals=0).astype(int)

        # calculate mask for current POI
        c_mask = mask_centers(corners, h, w, 15)

        # update POI with new locations
        c = c[c_mask] + vxs_whole
        r = r[c_mask] + vys_whole

        # calculate mask for next POI
        v_mask = mask_centers(np.vstack([r,c]).T, h, w, 15)

        # calculate the remainders for next interation
        vxs_rem += vxs - vxs_whole
        vys_rem += vys - vys_whole

        # mask the remainders that are valid for the next iteration
        vxs_rem = vxs_rem[v_mask]
        vys_rem = vys_rem[v_mask]

        # remove first image from the list and continue iterating
        imgs.pop(0)
    
    if save_vid:
        fps = 12
        fh, fw, *_ = fig_data[0].shape

        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        vname = f'video_{vid_name}.avi'
        video = cv2.VideoWriter(vname, fourcc, fps, (fw, fh))
        print(f"Writing {vname}")

        for f in fig_data:
            video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()

if __name__ == "__main__":
    vid_dir = None

    while vid_dir is None:
        vid_sel = input("Select video:\n1) Toy\n2) Doll\n[1/2]: ")
        if vid_sel == '1':
            vid_dir = "./images/toy"
        elif vid_sel == '2':
            vid_dir = "./images/doll"

    vid_name = vid_dir[vid_dir.rfind("/")+1:]
    imgs = [plt.imread(f"{vid_dir}/{i}") for i in sorted(os.listdir(vid_dir))]

    track_flow(imgs, vid_name, harris_thresh=1e-4, save_vid=False)
