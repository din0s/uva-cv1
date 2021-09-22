import numpy as np

def myPSNR(orig_image, approx_image):
    MSE = np.mean((orig_image - approx_image)**2)
    RMSE = np.sqrt(MSE)

    PSNR = 20 * np.log10(np.max(orig_image) / RMSE)

    return PSNR


if __name__ == "__main__":
    import cv2
    original = cv2.imread('images/image1.jpg').astype(np.float32)
    salt_pepper = cv2.imread('images/image1_saltpepper.jpg').astype(np.float32)

    psnr = myPSNR(original, salt_pepper)
    print(psnr)
