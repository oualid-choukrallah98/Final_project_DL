"""
Medical Image Captioning Dataset
PyTorch Dataset class for Indiana X-Ray dataset
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from preprocessing.text_preprocessing import CaptionTokenizer
from preprocessing.image_preprocessing import ImagePreprocessor


class MedicalCaptionDataset(Dataset):
    """Dataset for medical image captioning"""

    def __init__(
        self,
        data_csv,
        image_dir,
        tokenizer,
        image_transform,
        max_caption_length=100
    ):
        """
        Initialize dataset

        Args:
            data_csv: Path to CSV file with columns [uid, filename, findings, ...]
            image_dir: Directory containing images
            tokenizer: CaptionTokenizer instance
            image_transform: Image transformation function
            max_caption_length: Maximum caption length (including special tokens)
        """
        self.data = pd.read_csv(data_csv)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample

        Returns:
            dict with keys:
                - image: Tensor [3, H, W]
                - caption_ids: Tensor [max_length]
                - attention_mask: Tensor [max_length]
                - caption_length: int (actual length before padding)
                - image_id: str (for tracking)
                - caption_text: str (original caption)
        """
        row = self.data.iloc[idx]

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, row['filename'])

        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image_tensor = torch.zeros(3, 224, 224)

        # Get caption
        caption_text = str(row['findings'])

        # Tokenize caption
        caption_ids, attention_mask = self.tokenizer.encode(
            caption_text,
            max_length=self.max_caption_length,
            add_special_tokens=True
        )

        # Calculate actual length (before padding)
        caption_length = sum(attention_mask)

        return {
            'image': image_tensor,
            'caption_ids': torch.tensor(caption_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'caption_length': caption_length,
            'image_id': str(row['uid']),
            'filename': row['filename'],
            'caption_text': caption_text
        }


def create_dataloader(
    data_csv,
    image_dir,
    tokenizer,
    image_transform,
    batch_size=16,
    max_caption_length=100,
    shuffle=True,
    num_workers=4,
    drop_last=False
):
    """
    Create DataLoader for the dataset

    Args:
        data_csv: Path to data CSV file
        image_dir: Directory containing images
        tokenizer: CaptionTokenizer instance
        image_transform: Image transformation
        batch_size: Batch size
        max_caption_length: Maximum caption length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance
    """
    dataset = MedicalCaptionDataset(
        data_csv=data_csv,
        image_dir=image_dir,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_caption_length=max_caption_length
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True  # Faster GPU transfer
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MedicalCaptionDataset...")

    from preprocessing.text_preprocessing import load_tokenizer
    from preprocessing.image_preprocessing import get_train_transform

    # Load tokenizer
    tokenizer = load_tokenizer('data/vocab.json')
    print(f"✓ Loaded tokenizer (vocab size: {len(tokenizer)})")

    # Create image transform
    image_transform = get_train_transform(image_size=224, augment=False)
    print(f"✓ Created image transform")

    # Create dataset
    dataset = MedicalCaptionDataset(
        data_csv='data/train_data.csv',
        image_dir='data/images/images_normalized',
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_caption_length=100
    )
    print(f"✓ Created dataset with {len(dataset)} samples")

    # Test getting a single sample
    print("\nTesting single sample retrieval...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Caption IDs shape: {sample['caption_ids'].shape}")
    print(f"  Caption length: {sample['caption_length']}")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Caption text: {sample['caption_text'][:100]}...")

    # Test DataLoader
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        data_csv='data/train_data.csv',
        image_dir='data/images/images_normalized',
        tokenizer=tokenizer,
        image_transform=image_transform,
        batch_size=4,
        max_caption_length=100,
        shuffle=False,
        num_workers=0  # 0 for testing to avoid multiprocessing issues
    )
    print(f"✓ Created DataLoader with {len(dataloader)} batches")

    # Get one batch
    print("\nTesting batch loading...")
    batch = next(iter(dataloader))
    print(f"  Batch image shape: {batch['image'].shape}")
    print(f"  Batch caption_ids shape: {batch['caption_ids'].shape}")
    print(f"  Batch caption lengths: {batch['caption_length']}")

    print("\n✓ Dataset and DataLoader working correctly!")
