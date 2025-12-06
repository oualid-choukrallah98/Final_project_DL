"""
Example usage of the BLIP and Mamba models
"""
import torch
from models.blip_model import BLIPModel
from models.mamba_decoder import MambaDecoder
from transformers import BlipProcessor
from PIL import Image

# Example 1: Using BLIP Model
def example_blip():
    print("Example 1: Using BLIP Model")
    
    # Load model
    model = BLIPModel(use_lora=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Load image
    image = Image.open("data/chest-xrays-indiana-university/images/images_normalized/2_IM-0652-1001.dcm.png").convert("RGB")
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50
        )
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Generated caption: {caption}\n")


# Example 2: Using Mamba Decoder (Standard)
def example_mamba_standard():
    print("Example 2: Using Mamba Decoder (Standard)")
    
    # Load model
    model = MambaDecoder(
        vocab_size=30522,
        d_model=768,
        n_layers=6,
        use_rope=False
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Load image
    image = Image.open("data/chest-xrays-indiana-university/images/images_normalized/2_IM-0652-1001.dcm.png").convert("RGB")
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50
        )
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Generated caption: {caption}\n")


# Example 3: Using Mamba Decoder (RoPE)
def example_mamba_rope():
    print("Example 3: Using Mamba Decoder (RoPE)")
    
    # Load model
    model = MambaDecoder(
        vocab_size=30522,
        d_model=768,
        n_layers=6,
        use_rope=True
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Load image
    image = Image.open("data/chest-xrays-indiana-university/images/images_normalized/2_IM-0652-1001.dcm.png").convert("RGB")
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50
        )
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Generated caption: {caption}\n")


if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Run examples (comment out if you don't have data)
    # example_blip()
    # example_mamba_standard()
    # example_mamba_rope()
    
    print("Examples ready. Uncomment the function calls above to run them.")

