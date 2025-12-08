from torch.utils.data import Dataset
from PIL import Image


class ChestIUDataset(Dataset):
    def __init__(self, image_dir, report_df, processor):
        self.image_dir = image_dir
        self.report_df = report_df
        self.processor = processor
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for i in range(len(self.report_df)):
            samples.append({"image": self.image_dir + "/" + str(self.report_df['filename'].iloc[i]), "caption": self.report_df['findings'].iloc[i]})
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
            return_tensors="pt"
        )
        
        # Remove batch dimension added by processor
        return {k: v.squeeze(0) for k, v in encoding.items()}