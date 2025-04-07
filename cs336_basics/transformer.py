# add to path
import sys
sys.path.append('/users/christineye/cs336/assignment1-basics/cs336_basics')

import layers
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        # kwargs: device, dtype, etc.
        # transformer block = x + FFN(RMSNorm(x + attn(RMSNorm(x))))
        # can reuse RoPE any number of times
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.ln1 = layers.RMSNorm(d_model, **kwargs)
        self.attn = layers.MultiHeadSelfAttention(d_model, num_heads, rope = rope, **kwargs)
        self.ln2 = layers.RMSNorm(d_model, **kwargs)
        self.ffn = layers.SwiGLU(d_model, d_ff, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # it's so beautiful that my layers are all properly batched!
        h = x + self.attn(self.ln1(x))
        
        # i guess it's not a proper residual connection? literally so much guesswork to pass the tests lol
        # the x add is already implicit in h
        h = h + self.ffn(self.ln2(h))
        return h

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__()
        self.rope = layers.RoPE(rope_theta, d_model // num_heads, context_length)
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = layers.RMSNorm(d_model)
        # note that the weights are stored out_features, in_features
        self.lm_head = layers.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
