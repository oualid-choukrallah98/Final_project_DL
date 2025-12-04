import torch
import torch.nn as nn
from typing import Optional
import warnings
from mamba_ssm import Mamba

try:
    from rotary_embedding_torch import RotaryEmbedding
    ROTARY_EMBEDDING_AVAILABLE = True
except ImportError:
    ROTARY_EMBEDDING_AVAILABLE = False
    RotaryEmbedding = None



class MambaDecoder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        visual_feature_dim: int,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.2,  
        use_rope: bool = False,
        rope_base: int = 10000
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.visual_feature_dim = visual_feature_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if use_rope:
            if not ROTARY_EMBEDDING_AVAILABLE:
                warnings.warn("RoPE requested but rotary_embedding_torch not availble, using learned postional embeddings instead.")
                self.use_rope = False
                self.rope = None
                self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            else:
                self.rope = RotaryEmbedding(dim=d_model)
                self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.rope = None
        
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        caption_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = visual_features.shape[0]
        
        if visual_features.shape[1] > 1:
            visual_context = visual_features.mean(dim=1)
        else:
            visual_context = visual_features.squeeze(1)
        
        visual_context = self.visual_projection(visual_context)
        
        if caption_ids is None:
            return self._generate(visual_context, max_length=self.max_seq_len)
        
        seq_len = caption_ids.shape[1]
        
        token_embeds = self.token_embedding(caption_ids)
        
        if self.use_rope and self.rope is not None:
            token_embeds = self.rope.rotate_queries_or_keys(token_embeds)
        else:
            positions = torch.arange(seq_len, device=token_embeds.device)
            pos_embeds = self.pos_embedding(positions)
            token_embeds = token_embeds + pos_embeds.unsqueeze(0)
        
        visual_context = visual_context.unsqueeze(1)
        x = torch.cat([visual_context, token_embeds], dim=1)
        
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x)
            x = self.dropout(x)
            x = x + residual
        
        x = x[:, 1:]
        
        logits = self.output_projection(x)
        
        return logits
    
    def _generate(
        self,
        visual_context: torch.Tensor,
        max_length: int = 200,
        temperature: float = 1.0,
        bos_id: int = 1,
        eos_id: int = 2
    ) -> torch.Tensor:
        batch_size = visual_context.shape[0]
        device = visual_context.device
        
        generated_ids = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=device
        )
        
        visual_context = visual_context.unsqueeze(1)
        
        for step in range(max_length - 1):
            current_seq = generated_ids
            seq_len = current_seq.shape[1]
            
            token_embeds = self.token_embedding(current_seq)
            
            if self.use_rope and self.rope is not None:
                token_embeds = self.rope.rotate_queries_or_keys(token_embeds)
            else:
                positions = torch.arange(seq_len, device=device)
                pos_embeds = self.pos_embedding(positions)
                token_embeds = token_embeds + pos_embeds.unsqueeze(0)
            
            x = torch.cat([visual_context, token_embeds], dim=1)
            
            for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
                residual = x
                x = layer_norm(x)
                x = mamba_layer(x)
                x = x + residual
            
            last_hidden = x[:, -1, :]
            logits = self.output_projection(last_hidden)
            
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if (next_token == eos_id).all():
                break
        
        return generated_ids

