import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    This module pre-computes the cosine and sine frequency bands and applies
    them to the input tensor during the forward pass.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Constructs the RoPE module and creates the necessary buffers.

        Args:
            theta (float): The base for the geometric progression of frequencies.
            d_k (int): The dimension of the query and key vectors. Must be even.
            max_seq_len (int): The maximum sequence length that will be inputted.
            device (torch.device, optional): Device to store the buffer on.
                                              Defaults to None.
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k (dimension) must be an even number, but got {d_k}")

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute the frequency bands (the 'θ_k' term in the paper)
        # Formula from the paper: θ_{i,k} = i / (Θ^(2(k-1)/d))
        # We can pre-compute the denominator part: freqs = 1.0 / (theta ^ (2k'/d))
        # where k' is the 0-indexed dimension pair index
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        # Shape: (d_k / 2,)

        # Create position indices
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float)
        # Shape: (max_seq_len,)

        # Compute the full angle matrix using broadcasting: θ_{i,k} = i * freqs_k
        freqs_cis = torch.outer(t, freqs)
        # Shape: (max_seq_len, d_k / 2)

        # As per the prompt, register the pre-computed sin/cos values as buffers.
        # `persistent=False` means they won't be saved in the model's state_dict.
        self.register_buffer('cos_cached', torch.cos(freqs_cis), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs_cis), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor by applying RoPE.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
                              The '...' denotes any number of batch dimensions.
            token_positions (torch.Tensor): A tensor of shape (..., seq_len) that specifies
                                            the absolute position of each token in x.

        Returns:
            torch.Tensor: The output tensor with RoPE applied, having the same shape as x.
        """
        # Ensure the pre-computed buffers are on the same device as the input tensor.
        # PyTorch moves buffers with .to(device), but we ensure it here for safety.
        cos = self.cos_cached[token_positions].to(x.device)
        sin = self.sin_cached[token_positions].to(x.device)
        
        # The shapes of cos/sin will be (..., seq_len, d_k/2) due to indexing.
        # We need to reshape them to be broadcastable with the even/odd parts of x.
        # The `...` in token_positions must match the `...` in x, so direct
        # broadcasting should work as intended.

        # Split the input tensor `x` into its even and odd indexed dimensions.
        x_even = x[..., ::2]  # Shape: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # Shape: (..., seq_len, d_k/2)

        # Apply the rotation based on the 2D rotation matrix formula:
        # R * [x_even, x_odd]^T = [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]^T
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Create a new tensor to store the interleaved results.
        x_rotated = torch.empty_like(x)
        
        # Place the rotated parts back into the new tensor.
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated