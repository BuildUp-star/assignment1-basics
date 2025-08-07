import torch
import torch.nn as nn
from torch import device, dtype
from cs336_basics.linear import Linear
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor


class SwiGLU(nn.Module):
    """
    A position-wise feed-forward network using the SwiGLU activation.
    This implementation follows the Llama 2 architecture for its FFN layers.
    """
    def __init__(self, d_model: int, device: device | None = None, dtype: dtype | None = None):
        super().__init__()

        # 1. Calculate the intermediate dimension (d_ff) as per the instructions.
        # This is a common practice in modern LLMs like Llama.
        hidden_dim_approx = int(d_model * 8 / 3)
        # Ensure d_ff is a multiple of 64 for hardware efficiency.
        # This is an efficient way to round up to the nearest multiple.
        self.d_ff = ((hidden_dim_approx + 63) // 64) * 64
        
        factory_kwargs = {"device": device, "dtype": dtype}

        # 2. Define the three linear layers using your custom Linear class.
        self.w1 = Linear(d_model, self.d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the data flow through the SwiGLU FFN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        # Calculate the main path and the gate path in parallel
        main_path = self.w1(x)
        gate_path = self.w3(x)

        # Apply the SiLU activation to the main path and multiply by the gate
        # F.silu is a built-in, efficient version of x * sigmoid(x)
        gated = main_path * torch.sigmoid(main_path) * gate_path
        
        # Apply the final down-projection layer
        output = self.w2(gated)
        
        return output
    
def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # 1. Subtract the max for numerical stability.
    # torch.max returns a tuple of (values, indices), so we extract the values tensor.
    max_values = torch.max(in_features, dim=dim, keepdim=True).values
    centered_features = in_features - max_values
    
    # 2. Exponentiate.
    exp_features = centered_features.exp()
    
    # 3. Sum along the specified dimension to get the denominator.
    # torch.sum directly returns a tensor with the sum, so .values is not needed.
    sum_exp_features = torch.sum(exp_features, dim=dim, keepdim=True)
    
    # 4. Return the final result.
    return exp_features / sum_exp_features

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    res = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k))  # queries keys
    if mask is not None:
        res[mask == False] = float("-inf")  # Set masked positions to -inf for softmax
    res = softmax(res, dim=-1)  # Apply softmax to the last dimension
    res = res @ V  # Multiply by the values tensor
    return res