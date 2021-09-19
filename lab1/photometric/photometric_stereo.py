import numpy as np
import cv2
import os
from numpy.linalg import norm
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./images/SphereGray5/', rgb=False):
    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    if rgb:
        img_stack_rgb, scriptV_rgb = [], []
        for channel in range(3):
            [image_stack, scriptV] = load_syn_images(image_dir, channel)
            img_stack_rgb.append(image_stack)
            scriptV_rgb.append(scriptV)
        [h, w, n] = img_stack_rgb[0].shape
    else:
        [image_stack, scriptV] = load_syn_images(image_dir)
        [h, w, n] = image_stack.shape

    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    if rgb:
        albedo = np.empty((h, w, 3))
        normals_rgb = []
        for channel in range(3):
            [_albedo, _normals] = estimate_alb_nrm(img_stack_rgb[channel], scriptV_rgb[channel])
            _albedo = cv2.normalize(_albedo, None, 0.0, 1.0, cv2.NORM_MINMAX)
            albedo[:, :, 2 - channel] = _albedo
            normals_rgb.append(_normals)
        normals = np.mean(normals_rgb, axis=0)

        normals_norm = np.linalg.norm(normals, axis=2)
        for i in range(3):
            normals[:, :, i] /= normals_norm
        normals[normals != normals] = 0

    else:
        [albedo, normals] = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    # thresholds = np.logspace(-3.5, -1, 20)
    # outliers = [100 * np.sum(SE > t) / len(SE.flatten()) for t in thresholds]
    # plt.figure()
    # plt.grid()
    # plt.plot(thresholds, outliers, 'o-')
    # plt.plot([thresholds[7]], [outliers[7]], 'r.')
    # plt.annotate("%.4f" % thresholds[7], (thresholds[7] + 0.5 * 1e-3, outliers[7]))
    # plt.xlabel("threshold")
    # plt.ylabel("outliers %")
    # plt.xscale('log')
    # plt.show()
    threshold = 0.0026
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)

## Face
def photometric_stereo_face(image_dir='./images/yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, False)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q )

    # show results
    show_results(albedo, normals, height_map, SE)

if __name__ == '__main__':
    # photometric_stereo('./images/MonkeyColor/', True)
    photometric_stereo_face()
