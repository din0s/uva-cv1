from utils import image_derivatives, normal1chan, conv2d

import matplotlib.pyplot as plt
import numpy as np

def get_centers(h: int, w: int, stride: int = 15, return_plots: bool = False) -> np.ndarray:
    s = stride // 2                                     # integer division by 2 to get left/right window
    r = np.arange(s, h - s, step=stride)                # rows go from stride//2 to max_height - stride//2
    c = np.arange(s, w - s, step=stride)                # cols go from stride//2 to max_width - stride//2
    return np.array(np.meshgrid(r,c)).T.reshape(-1, 2)  # reshape to get (#centers, 2)

def blockify(img: np.ndarray, centers: np.ndarray, stride: int = 15) -> np.ndarray:
    s = stride // 2                                             # integer division by 2 to get left/right window
    blocks = [img[r-s:r+s+1, c-s:c+s+1] for (r,c) in centers]   # select stride//2 pixels to the left/right
    return np.array(blocks).reshape(len(centers), stride**2)    # reshape to get (#centers, stride**2)

def optical_flow(img0: np.ndarray, img1: np.ndarray, centers: np.ndarray = None,
                 stride: int = 15, blockPlot: bool = True, return_plots: bool = False):
    img1 = normal1chan(img1)            # make sure we have a normalized image with 1 channel
    Ix, Iy = image_derivatives(img1)    # calculate image derivatives dx, dy

    h, w = img1.shape
    if centers is None:
        # generate centers by evenly spacing them
        centers = get_centers(h, w, stride)
    else:
        s = stride // 2
        # drop centers that are outside the valid stride window on either side
        inside = (s <= centers[:, 0]) & (s <= centers[:, 1]) & (centers[:, 0] < h-s) & (centers[:, 1] < w-s)
        centers = centers[inside]

    # split the img derivatives into blocks
    Rx = blockify(Ix, centers, stride)
    Ry = blockify(Iy, centers, stride)

    As = np.dstack([Rx,Ry])                                             # A = [ Ix Iy ]
    bs = blockify(img1 - normal1chan(img0), centers)                    # b = [ -It ]
    vs = np.array([np.linalg.pinv(A) @ b for (A, b) in zip(As, bs)])    # v = pinv(A) @ b
    fig = plt.figure()
    # display first image
    plt.axis('off')
    plt.imshow(img0)

    R, C = centers[:, 0], centers[:, 1]
    MAGIC_NUMBER = np.linalg.norm(vs)
    MAGIC_NUMBER = 2.7 * np.tanh(MAGIC_NUMBER)
    print(MAGIC_NUMBER)
    vxs, vys = vs[:, 0] * MAGIC_NUMBER, vs[:, 1] * MAGIC_NUMBER
    # CAREFUL: cols correspond to vxs, rows correspond to vys
    plt.quiver(C, R, vxs, vys, angles="xy", scale_units="xy", scale=0.1, cmap='Reds')
    #plt.show(block=blockPlot)

    if not blockPlot:
        # we were asked to not block the plot display, which is done for the tracking feature
        # let's just wait a little bit (24fps) and clear the figure to draw the next frame
        plt.pause(1/24)
        plt.clf()

    plt.close()
    if return_plots:
        return R, C, vxs, vys, inside, fig
    else:
        return R, C, vxs, vys, inside
    # update the centers accordingly and return
    # C = C + np.round(vxs * 2).astype(int)
    # R = R + np.round(vys * 2).astype(int)

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
