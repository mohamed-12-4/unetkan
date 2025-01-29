import matplotlib.pyplot as plt
import numpy as np
def visualize_image(image_tensor, mask_tensor=None):
    """
    Visualizes an image and its corresponding mask.

    Args:
        image_tensor (torch.Tensor): The image tensor in CHW format.
        mask_tensor (torch.Tensor, optional): The mask tensor in CHW format.
    """
    # Convert image from CHW (C, H, W) to HWC (H, W, C)
    image = image_tensor.permute(1, 2, 0).numpy()  # C -> HWC

    # If image values are in [0, 1], scale to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Visualize the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    if mask_tensor is not None:
        # Convert mask to HWC format (for visualization)
        mask = mask_tensor.squeeze(0).numpy()  # Remove the channel dimension for mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis("off")

    plt.show()
