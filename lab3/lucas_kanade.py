from utils import image_derivatives, mask_centers, normal1chan

import matplotlib.pyplot as plt
import numpy as np

def get_centers(h: int, w: int, stride: int = 15) -> np.ndarray:
    s = stride // 2                                     # integer division by 2 to get left/right window
    r = np.arange(s, h - s, step=stride)                # rows go from stride//2 to max_height - stride//2
    c = np.arange(s, w - s, step=stride)                # cols go from stride//2 to max_width - stride//2
    return np.array(np.meshgrid(r,c)).T.reshape(-1, 2)  # reshape to get (#centers, 2)

def blockify(img: np.ndarray, centers: np.ndarray, stride: int = 15) -> np.ndarray:
    s = stride // 2                                             # integer division by 2 to get left/right window
    blocks = [img[r-s:r+s+1, c-s:c+s+1] for (r,c) in centers]   # select stride//2 pixels to the left/right
    return np.array(blocks).reshape(len(centers), stride**2)    # reshape to get (#centers, stride**2)

def optical_flow(img0: np.ndarray, img1: np.ndarray, centers: np.ndarray = None,
                 stride: int = 15, block_plot:bool = True, video_mode: bool = False) -> tuple:
    img1 = normal1chan(img1)            # make sure we have a normalized image with 1 channel
    Ix, Iy = image_derivatives(img1)    # calculate image derivatives dx, dy

    h, w = img1.shape
    if centers is None:
        # generate centers by evenly spacing them
        centers = get_centers(h, w, stride)
    else:
        # drop centers that are outside the valid stride window on either side
        c_mask = mask_centers(centers, h, w, stride)
        centers = centers[c_mask]

    # split the img derivatives into blocks
    Rx = blockify(Ix, centers, stride)
    Ry = blockify(Iy, centers, stride)

    As = np.dstack([Rx,Ry])                                             # A = [ Ix Iy ]
    bs = blockify(img1 - normal1chan(img0), centers)                    # b = [ -It ]
    vs = np.array([np.linalg.pinv(A) @ b for (A, b) in zip(As, bs)])    # v = pinv(A) @ b

    fig = plt.figure(0)
    plt.axis('off')
    plt.imshow(img0)

    R, C = centers[:, 0], centers[:, 1]
    
    if video_mode:
        SCALING_FACTOR = np.linalg.norm(vs)
        SCALING_FACTOR = 2.7 * np.tanh(SCALING_FACTOR)
    else:
        SCALING_FACTOR = 1

    vxs, vys = vs[:, 0] * SCALING_FACTOR, vs[:, 1] * SCALING_FACTOR

    # CAREFUL: cols correspond to vxs, rows correspond to vys
    plt.quiver(C, R, vxs, vys, angles="xy", scale_units="xy", scale=0.1, cmap='Reds')

    if video_mode:
        fig.canvas.draw()
        plt.close(fig)

        # also include the figure for video mode
        return vxs, vys, fig
    
    plt.show(block=block_plot) # not in video mode, showing plot normally
    if not block_plot:
        # we were asked to not block the plot display, which is done for the tracking feature
        # let's just wait a little bit (24fps) and clear the figure to draw the next frame
        plt.pause(1/24)
        plt.clf()

    return vxs, vys

if __name__ == "__main__":
    img_src = None

    while img_src is None:
        img_sel = input("Select image:\n1) Car\n2) Coke\n[1/2]: ")
        if img_sel == '1':
            img_src = "./images/Car%d.jpg"
        elif img_sel == '2':
            img_src = "./images/Coke%d.jpg"

    img0 = plt.imread(img_src % 1)
    img1 = plt.imread(img_src % 2)
    optical_flow(img0, img1)
