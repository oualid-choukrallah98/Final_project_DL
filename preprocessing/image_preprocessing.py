"""
Image Preprocessing Module
Handles image transformations for medical chest X-rays
"""

import torch
from torchvision import transforms
from PIL import Image


class ImagePreprocessor:
    """Preprocessor for chest X-ray images"""

    def __init__(self, image_size=224, is_training=True, augment=False):
        """
        Initialize image preprocessor

        Args:
            image_size: Target image size (height and width)
            is_training: Whether this is for training (affects normalization)
            augment: Whether to apply data augmentation (only for training)
        """
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment and is_training

        # ImageNet statistics (for pre-trained models like ViT/ResNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Build transformation pipeline
        self.transform = self._build_transform()

    def _build_transform(self):
        """Build transformation pipeline"""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        # Data augmentation (only for training)
        if self.augment:
            # Be careful with medical images - minimal augmentation
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),  # Chest X-rays can be flipped
                transforms.RandomRotation(degrees=5),     # Small rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Slight brightness/contrast
            ])

        # Convert to tensor (automatically scales to [0, 1])
        transform_list.append(transforms.ToTensor())

        # Normalize using ImageNet statistics
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        return transforms.Compose(transform_list)

    def __call__(self, image):
        """
        Apply transformations to image

        Args:
            image: PIL Image or path to image file

        Returns:
            transformed_image: Tensor of shape [3, image_size, image_size]
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)

        # Convert grayscale to RGB (repeat channel 3 times)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations
        return self.transform(image)


def get_train_transform(image_size=224, augment=True):
    """Get transformation for training data"""
    return ImagePreprocessor(image_size=image_size, is_training=True, augment=augment)


def get_val_transform(image_size=224):
    """Get transformation for validation/test data"""
    return ImagePreprocessor(image_size=image_size, is_training=False, augment=False)


if __name__ == "__main__":
    # Test the image preprocessor
    import os

    print("Testing ImagePreprocessor...")

    # Find a sample image
    image_dir = 'data/images/images_normalized'
    if os.path.exists(image_dir):
        sample_images = [f for f in os.listdir(image_dir) if f.endswith('.png')][:3]

        if sample_images:
            # Test training transform (with augmentation)
            train_transform = get_train_transform(image_size=224, augment=True)
            print(f"\nTesting training transform (with augmentation)...")

            for img_file in sample_images:
                img_path = os.path.join(image_dir, img_file)
                img_tensor = train_transform(img_path)
                print(f"  {img_file}: shape={img_tensor.shape}, "
                      f"min={img_tensor.min():.2f}, max={img_tensor.max():.2f}, "
                      f"mean={img_tensor.mean():.2f}")

            # Test validation transform (no augmentation)
            val_transform = get_val_transform(image_size=224)
            print(f"\nTesting validation transform (no augmentation)...")

            img_path = os.path.join(image_dir, sample_images[0])
            img_tensor = val_transform(img_path)
            print(f"  {sample_images[0]}: shape={img_tensor.shape}, "
                  f"min={img_tensor.min():.2f}, max={img_tensor.max():.2f}, "
                  f"mean={img_tensor.mean():.2f}")

            print("\n✓ Image preprocessor working correctly!")
        else:
            print(f"\n✗ No images found in {image_dir}")
    else:
        print(f"\n✗ Image directory not found: {image_dir}")
