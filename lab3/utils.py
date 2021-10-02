import numpy as np
import scipy.signal as sig

def conv2d(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    return sig.convolve2d(image, filter, mode='same', boundary='symm')

def normal2chan(img: np.ndarray) -> np.ndarray:
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

    # make sure the image is normalized and has 2 channels
    img = normal2chan(img)

    return conv2d(img, Gx), conv2d(img, Gy)
