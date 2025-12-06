"""
BLIP Model for Medical Image Captioning
Based on the notebook implementation with encoder + decoder
"""
import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import LoraConfig, get_peft_model
from transformers.models.blip import modeling_blip_text


def apply_blip_fix():
    """Apply the fixed forward function for BLIP embeddings"""
    def fixed_forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            if position_ids is None:
                if input_ids is not None:
                    seq_length = input_ids.shape[1]
                    position_ids = torch.arange(
                        past_key_values_length, 
                        seq_length + past_key_values_length, 
                        dtype=torch.long, 
                        device=embeddings.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[:2])
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings  # Out-of-place addition

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    modeling_blip_text.BlipTextEmbeddings.forward = fixed_forward


class BLIPModel(nn.Module):
    """BLIP Model wrapper for medical image captioning"""
    
    def __init__(self, use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        # Apply the BLIP fix
        apply_blip_fix()
        
        # Load base model
        self.base_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True
        )
        
        # Enable gradients for input embeddings
        self.base_model.enable_input_require_grads()
        
        if use_lora:
            # Configure LoRA
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none"
            )
            self.model = get_peft_model(self.base_model, config)
        else:
            self.model = self.base_model
    
    def forward(self, input_ids, pixel_values, labels=None, attention_mask=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def generate(self, pixel_values, max_new_tokens=50, **kwargs):
        """Generate captions from images"""
        inputs = {"pixel_values": pixel_values}
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    
    def save_pretrained(self, path):
        """Save the model"""
        self.model.save_pretrained(path)
    
    def load_pretrained(self, path):
        """Load a saved model"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        return self
    
    def print_trainable_parameters(self):
        """Print trainable parameters"""
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable:,} || All params: {total:,} || Trainable%: {100 * trainable / total:.2f}")

