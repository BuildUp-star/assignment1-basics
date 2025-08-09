import torch
from torch import nn
from torch import Tensor, device, dtype

from cs336_basics.utils import Linear, softmax, scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding as RoPE
from jaxtyping import Float, Int

import torch
import torch.nn as nn
import math
from typing import Optional

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Implements a Causal Multi-Head Self-Attention mechanism with Rotary
    Positional Embeddings (RoPE).

    This module projects an input tensor `x` into Query (Q), Key (K), and
    Value (V) tensors using a single combined linear layer for efficiency.
    It then applies RoPE to the Q and K tensors, computes attention scores
    using a causal mask, and finally concatenates the outputs from multiple
    heads before a final linear projection.
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Initializes the CausalMultiHeadSelfAttention module.

        Args:
            d_model (int): The total dimensionality of the input and output features.
            num_heads (int): The number of parallel attention heads to use.
            device (torch.device, optional): The device to place the weight tensors on.
            dtype (torch.dtype, optional): The data type for the weight tensors.
        """
        super().__init__()
        # d_model must be divisible by num_heads to ensure an even split.
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        # d_k is the dimension of each individual attention head.
        self.d_k = d_model // num_heads

        # Keyword arguments for device and dtype to pass to the Linear layer constructor.
        factory_kwargs = {"device": device, "dtype": dtype}

        # A single, combined linear layer for Q, K, and V projections.
        # This is more efficient than three separate matrix multiplications.
        # It projects the input from d_model to 3 * d_model to hold Q, K, and V.
        self.W_qkv = Linear(d_model, 3 * d_model, **factory_kwargs)

        # The final linear layer to project the concatenated attention head outputs
        # back to the model's dimension (d_model).
        self.W_o = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
    ) -> torch.Tensor:
        """
        Performs the forward pass for causal multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor.
                              Shape: (B, T, C), where B is batch_size, T is sequence_length,
                              and C is d_model. e.g., (2, 128, 64)
            token_positions (torch.Tensor): The absolute positions of tokens in the sequence.
                                            Shape: (B, T). e.g., (2, 128)
            theta (float): The base period for RoPE.
            max_seq_len (int): The maximum sequence length for pre-computing RoPE caches.

        Returns:
            torch.Tensor: The output tensor of the same shape as the input `x`.
        """
        # B: batch_size, T: sequence_length, C: d_model
        B, T, C = x.shape

        # 1. Project to Q, K, V in a single pass.
        # Shape change: (B, T, C) -> (B, T, 3 * C)
        # e.g., (2, 128, 64) -> (2, 128, 192)
        qkv = self.W_qkv(x)

        # Split the 3*C dimension into three separate tensors for Q, K, and V.
        # Shape change: (B, T, 3 * C) -> three tensors of shape (B, T, C)
        # e.g., (2, 128, 192) -> three of (2, 128, 64)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 2. Reshape Q, K, V for Multi-Head processing.
        # First, view the last dimension C as (num_heads, d_k).
        # Shape change (view): (B, T, C) -> (B, T, num_heads, d_k)
        # e.g., (2, 128, 64) -> (2, 128, 4, 16)
        #
        # Then, transpose the sequence length (T) and num_heads dimensions.
        # This makes num_heads a "batch-like" dimension for efficient, parallel computation.
        # Shape change (transpose): (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        # e.g., (2, 128, 4, 16) -> (2, 4, 128, 16)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings (RoPE) to Q and K.
        # This is only done if token positions are provided.
        if token_positions is not None:
            rope = RoPE(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len, device=x.device)
            # Unsqueeze token_positions to make it broadcastable across the num_heads dimension.
            # Shape change: (B, T) -> (B, 1, T)
            # This allows it to work with q and k of shape (B, num_heads, T, d_k).
            # We pass `q` and `k` as the first positional argument `x`.
            q = rope(q, token_positions=token_positions.unsqueeze(1))
            k = rope(k, token_positions=token_positions.unsqueeze(1))
        # If no positions are provided, q and k remain unchanged.

        # 4. Create the causal mask to prevent attention to future tokens.
        # torch.triu creates an upper-triangular matrix. diagonal=1 ensures the
        # main diagonal is all False (a token can attend to itself).
        # `True` values in the mask indicate positions that should be ignored.
        # Shape: (T, T), e.g., (128, 128)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # 5. Perform scaled dot-product attention.
        # The utility function is expected to handle broadcasting the 2D causal_mask
        # across the 4D attention score tensor (B, num_heads, T, T).
        # Output shape is the same as V: (B, num_heads, T, d_k)
        # e.g., (2, 4, 128, 16)
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 6. Concatenate heads and reshape back to the original tensor shape.
        # First, transpose back: (B, num_heads, T, d_k) -> (B, T, num_heads, d_k)
        # e.g., (2, 4, 128, 16) -> (2, 128, 4, 16)
        # .contiguous() is required to ensure the tensor is in a contiguous block
        # of memory before calling .view().
        # Then, view to merge the last two dimensions: (B, T, num_heads, d_k) -> (B, T, C)
        # e.g., (2, 128, 4, 16) -> (2, 128, 64)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Apply the final linear projection.
        # Shape change: (B, T, C) -> (B, T, C) (no shape change)
        # e.g., (2, 128, 64) -> (2, 128, 64)
        output = self.W_o(attn_output)

        return output

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.utils import SwiGLU

class TransformerBlock(nn.Module):
    """
    A complete Transformer block implementing a Pre-Norm architecture with
    Causal Multi-Head Self-Attention and a SwiGLU Feed-Forward Network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # Layer Normalization before the attention block
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        # Our fully implemented MHA module
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        # Layer Normalization before the FFN block
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Instantiate the SwiGLU module directly
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor, 
        theta: float, 
        max_seq_len: int
    ) -> torch.Tensor:
        # First residual connection: Attention
        # Pre-Norm: Apply LayerNorm before passing to the attention module
        attn_input = self.ln1(x)
        attn_output = self.attn(attn_input, token_positions, theta, max_seq_len)
        x = x + attn_output

        # Second residual connection: FFN
        # Pre-Norm: Apply LayerNorm before passing to the FFN
        ffn_input = self.ln2(x)
        # A single, clean call to the FFN module
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        return x