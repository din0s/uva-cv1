import numpy as np
import cv2
import matplotlib.pyplot as plt
from iid_image_formation import reconstruct_image

def recolour_image(image_albedo, image_shading, initial_image):
	recoloured_image = np.zeros_like(initial_image)
	image_albedo = np.where(image_albedo!=0, 255, image_albedo) 
	image_albedo[..., 0] = 0
	image_albedo[..., 2] = 0
	recoloured_image = reconstruct_image(image_albedo, image_shading)
	return recoloured_image, image_albedo

def visualize(initial_image, image_albedo, image_shading, recoloured_image):
    fig, axs = plt.subplots(1, 4, figsize=(15, 15))

    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image')

    axs[1].imshow(image_albedo)
    axs[1].set_title('Image Albedo')

    axs[2].imshow(image_shading)
    axs[2].set_title('Image shading')


    axs[3].imshow(recoloured_image)
    axs[3].set_title('Recoloured Image')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

if __name__ == '__main__':
    image_albedo = cv2.imread('ball_albedo.png')[..., ::-1]
    image_shading = cv2.imread('ball_shading.png')[..., ::-1]
    initial_image = cv2.imread('ball.png')[..., ::-1]

    recoloured_image, recoloured_albedo = recolour_image(image_albedo, image_shading, initial_image)

    visualize(initial_image, recoloured_albedo, image_shading, recoloured_image)	
    