import cs336_basics.layers as layers
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    """Basic Transformer block."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        """Initialize a Transformer block.
        
        d_model: dimension of the model
        num_heads: number of attention heads
        d_ff: dimension of the feed-forward network
        rope: RoPE module, should be pre-initialized
        kwargs: device, dtype, etc.
        """

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
        """Forward pass of the Transformer block. Computes x + FFN(RMSNorm(x + attn(RMSNorm(x)))).
        
        x: input tensor
        """

        # it's so beautiful that my layers are all properly batched!
        h = x + self.attn(self.ln1(x))
        
        # the x add is already implicit in h
        h = h + self.ffn(self.ln2(h))
        return h

class TransformerLM(nn.Module):
    """Full Transformer language model."""
    
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        """Initialize a Transformer language model.
        
        d_model: dimension of the model
        vocab_size: size of the vocabulary
        context_length: context length
        num_layers: number of layers
        rope_theta: RoPE theta
        num_heads: number of attention heads
        d_ff: hidden dimension of the feed-forward network
        kwargs: device, dtype, etc.
        """

        super().__init__()
        self.rope = layers.RoPE(rope_theta, d_model // num_heads, context_length, **kwargs)
        self.embedding = layers.Embedding(vocab_size, d_model, **kwargs)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = layers.RMSNorm(d_model, **kwargs)
        # note that the weights are stored out_features, in_features
        self.lm_head = layers.Linear(d_model, vocab_size, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.layers(h)
        h = self.ln_final(h)
        h = self.lm_head(h)
        return h

"""Ablation Study: removing RMSNorm"""
class TransformerBlockNoRMSNorm(TransformerBlock):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        super().__init__(d_model, num_heads, d_ff, rope, **kwargs)

    # we only need to override the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(x)
        h = h + self.ffn(h)
        return h

class TransformerLMNoRMSNorm(TransformerLM):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__(d_model, vocab_size, context_length, num_layers, rope_theta, num_heads, d_ff, **kwargs)
        
        # rebuild blocks with no RMSNorm
        self.layers = nn.Sequential(*[
            TransformerBlockNoRMSNorm(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.layers(h)
        h = self.lm_head(h)
        # also no final layer norm
        return h


"""Ablation Study: post-norm vs pre-norm"""
class TransformerBlockPostNorm(TransformerBlock):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        super().__init__(d_model, num_heads, d_ff, rope, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # only override the forward pass
        h = self.ln1(x + self.attn(x))
        h = self.ln2(h + self.ffn(h))
        return h

class TransformerLMPostNorm(TransformerLM):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__(d_model, vocab_size, context_length, num_layers, rope_theta, num_heads, d_ff, **kwargs)
        # replace the blocks with post-norm
        self.layers = nn.Sequential(*[
            TransformerBlockPostNorm(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])


"""Ablation Study: removing positional embeddings"""
class TransformerLMNoPE(TransformerLM):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__(d_model, vocab_size, context_length, num_layers, rope_theta, num_heads, d_ff, **kwargs)
        
        self.rope = None
        # rebuild blocks with no positional embeddings
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])

"""Ablation Study: SiLU vs. SwiGLU"""
class TransformerBlockSiLU(TransformerBlock):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        super().__init__(d_model, num_heads, d_ff, rope, **kwargs)
        # override d_ff with d_model * 4
        self.ffn = layers.SiLU(d_model, d_model * 4, **kwargs)

class TransformerLMSiLU(TransformerLM):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__(d_model, vocab_size, context_length, num_layers, rope_theta, num_heads, d_ff, **kwargs)
        self.layers = nn.Sequential(*[
            TransformerBlockSiLU(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])

"""Ablation study: weight tying"""
class TransformerBlockWeightTying(TransformerBlock):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module = None, **kwargs):
        super().__init__(d_model, num_heads, d_ff, rope, **kwargs)
        # add second additional layer norms
        self.ln3 = layers.RMSNorm(d_ff, **kwargs)
        self.ln4 = layers.RMSNorm(d_ff, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.ln3(self.attn(self.ln1(x)))
        h = h + self.ln4(self.ffn(self.ln2(h)))
        return h

class TransformerLMWeightTying(TransformerLM):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, rope_theta: float, num_heads: int, d_ff: int, **kwargs):
        super().__init__(d_model, vocab_size, context_length, num_layers, rope_theta, num_heads, d_ff, **kwargs)

        # rebuild blocks with double layer norms
        self.layers = nn.Sequential(*[
            TransformerBlockWeightTying(d_model, num_heads, d_ff, self.rope, **kwargs)
            for _ in range(num_layers)
        ])

        # implement weight tying
        embedding_params = nn.Parameter(torch.zeros(vocab_size, d_model, **kwargs))

        # initialize
        stdev = (1 / (vocab_size + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(embedding_params, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)

        # tie weights, forward pass should be taken care of
        self.embedding.matrix = embedding_params
        self.lm_head.weight = embedding_params
    
