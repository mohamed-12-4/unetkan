import torch
import cv2
from torchvision import transforms
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        mask_path = self.df.iloc[idx, 1]



        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            # Handle the error, e.g., skip this image or raise an exception
            return None, None  # Or raise an exception
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Reshape the image using cv2.resize
        image = cv2.resize(image, (256, 256))

        # Ensure the image has 3 channels
        if image.shape[-1] != 3:
           image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Reshape the mask
        mask = cv2.resize(mask, (256, 256))

        # Convert to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask