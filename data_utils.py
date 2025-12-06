"""
Data utilities for medical image captioning
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor
import torch


class ChestIUDataset(Dataset):
    """Dataset for chest X-ray image captioning"""
    
    def __init__(self, image_dir, report_df, processor, max_length=512):
        self.image_dir = image_dir
        self.report_df = report_df
        self.processor = processor
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for i in range(len(self.report_df)):
            samples.append({
                "image": f"{self.image_dir}/{self.report_df['filename'].iloc[i]}",
                "caption": self.report_df['findings'].iloc[i]
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image"]).convert("RGB")
        caption = item["caption"]

        # Preprocess using BLIP's processor
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Ensure attention_mask is included
        if "attention_mask" not in encoding:
            # Create attention mask from input_ids (1 for non-padding, 0 for padding)
            input_ids = encoding["input_ids"]
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id or 0
            attention_mask = (input_ids != pad_token_id).long()
            encoding["attention_mask"] = attention_mask
        
        # Remove batch dimension added by processor
        return {k: v.squeeze(0) for k, v in encoding.items()}


def get_data_loader(
    image_dir: str,
    csv_path: str,
    processor,
    batch_size: int = 4,
    shuffle: bool = True,
    data_fraction: float = 1.0,
    max_length: int = 512
) -> DataLoader:
    """Create a DataLoader from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Apply data fraction
    if data_fraction < 1.0:
        n_samples = int(len(df) * data_fraction)
        df = df.head(n_samples)
    
    dataset = ChestIUDataset(image_dir, df, processor, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    return dataloader, df

