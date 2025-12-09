"""
Mamba Decoder for Medical Image Captioning
Two versions: Standard positional encoding and RoPE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipProcessor, BlipImageProcessor
from transformers.models.blip.modeling_blip import BlipVisionModel
import math
import json
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False



class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MambaBlock(nn.Module):
    """Mamba block with optional RoPE"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_rope=False, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.use_rope = use_rope

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: Simple linear layer (not true Mamba, but allows code to run)
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.SiLU(),
                nn.Linear(d_model * expand, d_model),
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryEmbedding(d_model // 2)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        if self.use_rope and hasattr(self, 'rope'):
            seq_len = x.shape[1]
            cos, sin = self.rope(seq_len, x.device)
            # Apply RoPE to the input (simplified version)
            x_rot = x
            x_rot_reshaped = x_rot.view(*x_rot.shape[:-1], -1, 2)
            x_rot_reshaped = torch.stack([
                x_rot_reshaped[..., 0] * cos.unsqueeze(0).unsqueeze(0) - 
                x_rot_reshaped[..., 1] * sin.unsqueeze(0).unsqueeze(0),
                x_rot_reshaped[..., 0] * sin.unsqueeze(0).unsqueeze(0) + 
                x_rot_reshaped[..., 1] * cos.unsqueeze(0).unsqueeze(0)
            ], dim=-1)
            x = x_rot_reshaped.view(*x.shape)
        
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaDecoder(nn.Module):
    """Mamba Decoder for image captioning"""
    
    def __init__(
        self,
        d_model=768,
        n_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        max_seq_len=512,
        use_rope=False,
        vision_model_name="Salesforce/blip-image-captioning-base",
        vocab_path=None,
        dropout=0.1,
        freeze_vision_encoder=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        # Vision encoder (BLIP vision model)
        self.vision_model = BlipVisionModel.from_pretrained(vision_model_name)
        vision_dim = self.vision_model.config.hidden_size

        # Freeze vision encoder to prevent overfitting
        if freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Project vision features to decoder dimension with dropout
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.Dropout(dropout)
        )

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.vocab_size = vocab_data['vocab_size']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}

        # Store special token IDs
        self.bos_token_id = self.word2idx.get('<BOS>', 1)
        self.eos_token_id = self.word2idx.get('<EOS>', 2)
        self.pad_token_id = self.word2idx.get('<PAD>', 0)
        self.unk_token_id = self.word2idx.get('<UNK>', 3)

        # Initialize tokenizer (None for custom vocab)
        self.tokenizer = None

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # Positional encoding
        if not use_rope:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, max_seq_len, d_model) * 0.02
            )
        else:
            self.pos_embedding = None

        # Mamba layers with increased dropout
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, use_rope=use_rope, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def tokenize_text(self, text):
        """Tokenize text using custom vocab or HuggingFace tokenizer"""
        import re

        if self.tokenizer is not None:
            # Use HuggingFace tokenizer
            return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        else:
            # Use custom vocabulary tokenization
            text = str(text).lower()
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = re.findall(r'\w+|[.,!?;]', text)

            # Convert to IDs
            ids = [self.bos_token_id]
            for token in tokens:
                ids.append(self.word2idx.get(token, self.unk_token_id))
            ids.append(self.eos_token_id)

            return torch.tensor([ids])

    def decode_ids(self, token_ids):
        """Decode token IDs back to text"""
        if self.tokenizer is not None:
            # Use HuggingFace tokenizer
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # Use custom vocabulary
            tokens = []
            for idx in token_ids:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                word = self.idx2word.get(idx, '<UNK>')
                if word not in ['<BOS>', '<EOS>', '<PAD>']:
                    tokens.append(word)
            return ' '.join(tokens)
    
    def forward(self, pixel_values, input_ids=None, labels=None, attention_mask=None):
        batch_size = pixel_values.shape[0]
        
        # Encode vision
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [B, N, vision_dim]
        vision_features = self.vision_proj(vision_features)  # [B, N, d_model]
        
        # Get text embeddings with dropout
        if input_ids is not None:
            text_embeds = self.embedding(input_ids)  # [B, L, d_model]
            text_embeds = self.embedding_dropout(text_embeds)
            seq_len = text_embeds.shape[1]
        else:
            # For generation, start with vision features
            seq_len = vision_features.shape[1]
            text_embeds = vision_features
        
        # Concatenate vision and text features
        # For simplicity, we'll use vision features as initial hidden state
        # and text embeddings for the sequence
        if input_ids is not None:
            # Combine vision and text
            # Use vision features as context and text as sequence
            x = text_embeds
            # Add vision context (mean pool or use first token)
            vision_context = vision_features.mean(dim=1, keepdim=True)  # [B, 1, d_model]
            x = torch.cat([vision_context, x], dim=1)  # [B, 1+L, d_model]
        else:
            x = vision_features
        
        # Add positional encoding
        if not self.use_rope:
            if self.pos_embedding is not None:
                pos_len = x.shape[1]
                if pos_len <= self.max_seq_len:
                    x = x + self.pos_embedding[:, :pos_len, :]
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output_dropout(x)

        # Get logits
        logits = self.lm_head(x)
        
        # For training, we need to align with labels
        if labels is not None:
            # If we added vision context, remove it from logits
            # logits shape: [B, 1+L, vocab_size] if vision context was added
            # labels shape: [B, L]
            if input_ids is not None:
                # Remove vision context token (first token) from logits
                # Now logits is [B, L, vocab_size] matching labels [B, L]
                logits_text = logits[:, 1:, :]  # Remove first token (vision context)
            else:
                logits_text = logits
            
            # Shift logits and labels for language modeling
            # Predict next token: logits[i] predicts labels[i+1]
            shift_logits = logits_text[..., :-1, :].contiguous()  # [B, L-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous()  # [B, L-1]
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
    
    def generate(self, pixel_values, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.9):
        """Generate captions from images"""
        self.eval()
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # Encode vision
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state
        vision_features = self.vision_proj(vision_features)

        # Start with vision features (mean pool for initial context)
        vision_context = vision_features.mean(dim=1, keepdim=True)  # [B, 1, d_model]

        # Get start token ID
        if self.tokenizer is not None:
            # Using HuggingFace tokenizer
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                start_token_id = self.tokenizer.bos_token_id
            elif hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                start_token_id = self.tokenizer.pad_token_id
            else:
                start_token_id = 0
        else:
            # Using custom vocabulary
            start_token_id = self.bos_token_id

        # Initialize with start token
        input_ids = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=device
        )

        generated_ids = input_ids.clone()

        # Generate tokens
        for _ in range(max_new_tokens):
            # Get embeddings for all generated tokens so far
            text_embeds = self.embedding(generated_ids)  # [B, seq_len, d_model]

            # Always prepend vision context to maintain image information
            x = torch.cat([vision_context, text_embeds], dim=1)  # [B, 1+seq_len, d_model]

            # Add positional encoding
            if not self.use_rope and self.pos_embedding is not None:
                pos_len = x.shape[1]
                if pos_len <= self.max_seq_len:
                    x = x + self.pos_embedding[:, :pos_len, :]

            # Apply Mamba layers
            for layer in self.layers:
                x = layer(x)

            x = self.norm(x)

            # Get logits for last token
            logits = self.lm_head(x[:, -1:, :])  # [B, 1, vocab_size]
            logits = logits.squeeze(1) / temperature  # [B, vocab_size]

            # Apply top-k and top-p filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for end token
            if self.tokenizer is not None:
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break
            else:
                # Custom vocabulary
                if (next_token == self.eos_token_id).all():
                    break

        return generated_ids

