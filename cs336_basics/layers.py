import torch
import torch.nn as nn
import einops
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype)) # store W^T
        stdev = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume x is a ... x input_features tensor
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.matrix = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device = device, dtype = dtype))
        torch.nn.init.trunc_normal_(self.matrix, mean = 0, std = 1, a = -3, b = 3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ensure types are correct
        one_hot = nn.functional.one_hot(token_ids, num_classes = self.vocab_size)
        one_hot = one_hot.to(self.matrix.dtype)
        return one_hot @ self.matrix

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input tensor: shape (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        result = None
        # get squared sums
        squared_sums = einops.reduce(x ** 2, "batch sequence d_model -> batch sequence 1", "sum")
        rms = torch.sqrt((1/self.d_model) * squared_sums + self.eps)

        gain_values = einops.rearrange(self.gain, "d_model -> 1 1 d_model")
        result = (x / rms) * gain_values
        
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device = None, dtype = None):
        super().__init__()
        
        if d_ff is None:
            d_ff = (8 / 3) * d_model
            d_ff = 64 * math.ceil(d_ff / 64)
        
        self.W1 = nn.Parameter(torch.zeros(d_ff, d_model, device = device, dtype = dtype))
        self.W2 = nn.Parameter(torch.zeros(d_model, d_ff, device = device, dtype = dtype))
        self.W3 = nn.Parameter(torch.zeros(d_ff, d_model, device = device, dtype = dtype))
        stdev = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.W1, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.W2, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.W3, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einops.einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einops.einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        
        # element-wise products
        silu = w1x * torch.sigmoid(w1x)
        silu = silu * w3x

        result = einops.einsum(silu, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return result

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # compute cos and sin values
        # theta_i_k has shape (max_seq_len, d_k // 2)
        idx = torch.arange(0, max_seq_len, device = device, dtype = torch.float32)
       
        # note that it's 0-indexed even though the formula says 1-indexed in the handout!
        denom = theta ** (torch.arange(0, d_k, 2, device = device, dtype = torch.float32) / d_k)
        theta_i_k = idx.unsqueeze(1) / denom.unsqueeze(0)
        
        cos_cache = torch.cos(theta_i_k)
        sin_cache = torch.sin(theta_i_k)
        self.register_buffer("cos_cache", cos_cache, persistent = False)
        self.register_buffer("sin_cache", sin_cache, persistent = False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k % 2 == 0
        assert d_k == self.d_k

        # get cos/sin values up to sequence length
        if token_positions:
            cos_values = self.cos_cache[token_positions, :]
            sin_values = self.sin_cache[token_positions, :]
        else:
            cos_values = self.cos_cache[:seq_len, :]
            sin_values = self.sin_cache[:seq_len, :]

        # split into even and odd indices
        x_split = einops.rearrange(x, "... seq_len (d_split pair) -> ... seq_len d_split pair", d_split = self.d_k // 2, pair = 2)
        even_x = x_split[..., 0] # shape: (..., seq_len, d_k // 2)
        odd_x = x_split[..., 1] # shape: (..., seq_len, d_k // 2)

        # compute top and bottom halves of rotation
        x_rotate_even = even_x * cos_values - odd_x * sin_values
        x_rotate_odd = even_x * sin_values + odd_x * cos_values

        # stack back together
        x_rotated = torch.stack([x_rotate_even, x_rotate_odd], dim = -1)
        
        # implicitly interleaves the even and odd indices
        x_rotated = einops.rearrange(x_rotated, "... seq_len d_split pair -> ... seq_len (d_split pair)", d_split = self.d_k // 2, pair = 2)
        
        return x_rotated

# need to exp over all dimensions, but only apply softmax along the desired dimension
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # get max value over dimension dim
    offset = torch.max(x, dim = dim, keepdim = True)[0]

    # subtract offset
    x = x - offset

    # apply softmax along dimension dim
    x = torch.exp(x)
    # not everything needs to be einops
    x /= torch.sum(x, dim = dim, keepdim = True)
    
    return x

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    
    # sum over hidden dimension to get dot product
    attn_weights = einops.einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    attn_weights /= math.sqrt(d_k)
    
    # apply mask to last two dimensions, using -inf to zero out invalid positions
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
    
    # apply softmax over key dimension to get probability distribution over keys
    smax = softmax(attn_weights, dim = -1)

    # matmul with values, summing over key dimension (= apply attn weights to get weighted sum)
    result = einops.einsum(smax, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module, device = None, dtype = None):
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = rope

        # assuming d_k = d_v = d_model / num_heads
        super().__init__()
        # hdk, dmodel
        self.WQ = nn.Parameter(torch.zeros(d_model, d_model, device = device, dtype = dtype))
        self.WK = nn.Parameter(torch.zeros(d_model, d_model, device = device, dtype = dtype))
        # hdv, dmodel
        self.WV = nn.Parameter(torch.zeros(d_model, d_model, device = device, dtype = dtype))
        # dmodel, hdv
        self.WO = nn.Parameter(torch.zeros(d_model, d_model, device = device, dtype = dtype))
        
        # initialize weights with truncated normal
        stdev = (2 / (d_model + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.WQ, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.WK, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.WV, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.WO, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # dim 0 = queries, dim 1 = keys
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device, dtype = torch.float32))
        
        # multi-head attention; first compute vectors
        Q = einops.einsum(self.WQ, x, "(h dk) d_model, ... seq_len d_model -> ... h seq_len dk", h = self.num_heads)
        K = einops.einsum(self.WK, x, "(h dk) d_model, ... seq_len d_model -> ... h seq_len dk", h = self.num_heads)
        V = einops.einsum(self.WV, x, "(h dv) d_model, ... seq_len d_model -> ... h seq_len dv", h = self.num_heads)

        # now apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # compute dot product attention
        attn = scaled_dot_product_attention(Q, K, V, mask = causal_mask)
        attn = einops.rearrange(attn, "... h seq_len dv -> ... seq_len (h dv)", h = self.num_heads)

        # compute out projection
        output = einops.einsum(attn, self.WO, "... seq_len hdv, d_model hdv -> ... seq_len d_model")

        return output
