import numpy as np
import cv2
import matplotlib.pyplot as plt

def reconstruct_image(image_albedo, image_shading):
	# Convert to uint16 so you can multiply them
	reconstructed_image = np.multiply(image_shading.astype(np.uint16), image_albedo.astype(np.uint16))
	# Divide by 255 to roll back to the initial dimensions, convert to uint8
	return (reconstructed_image / 255).astype(np.uint8)

def visualize(initial_image, image_albedo, image_shading, reconstructed_image):
    fig, axs = plt.subplots(1, 4, figsize=(15, 15))

    axs[0].imshow(initial_image)
    axs[0].set_title('Initial Image')

    axs[1].imshow(image_albedo)
    axs[1].set_title('Image Albedo')

    axs[2].imshow(image_shading)
    axs[2].set_title('Image shading')

    axs[3].imshow(reconstructed_image)
    axs[3].set_title('Reconstructed Image')

    fig.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

if __name__ == '__main__':
    image_albedo = cv2.imread('ball_albedo.png')[..., ::-1]
    image_shading = cv2.imread('ball_shading.png')[..., ::-1]
    initial_image = cv2.imread('ball.png')[..., ::-1]

    reconstructed_image = reconstruct_image(image_albedo, image_shading)

    visualize(initial_image, image_albedo, image_shading, reconstructed_image)