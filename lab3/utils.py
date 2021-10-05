import numpy as np
import scipy.signal as sig

def conv2d(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    return sig.convolve2d(image, filter, mode='same', boundary='symm')

def normal1chan(img: np.ndarray) -> np.ndarray:
    dims = len(img.shape)
    assert dims == 2 or dims == 3

    if dims == 3:
        # take the mean of all channels
        img = np.mean(img, axis=2)

    if np.max(img) > 1:
        # normalize to [0,1]
        img = img / np.max(img)

    return img

def image_derivatives(img: np.ndarray) -> tuple:
    Gx = np.array([-1, 0, 1])[np.newaxis, :]
    Gy = Gx.T

    # make sure the image is normalized and has 1 channel
    img = normal1chan(img)

    return conv2d(img, Gx), conv2d(img, Gy)

def mask_centers(centers: np.ndarray, h: int, w: int, stride: int) -> np.ndarray:
    s = stride // 2
    return (s <= centers[:, 0]) & (s <= centers[:, 1]) & (centers[:, 0] < h-s) & (centers[:, 1] < w-s)
