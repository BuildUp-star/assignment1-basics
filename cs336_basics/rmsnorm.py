import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Initialize the RMSNorm layer.

        Args:
            d_model (int): Dimension of the model.
            eps (float): Small value to avoid division by zero.
            device: Device to place the parameters on (optional).
            dtype: Data type for the parameters (optional).
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Ensure input is float32 for numerical stability
        x_float = x.to(torch.float32)
        # Compute the root mean square of the last dimension
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale by weight
        return x_float / rms * self.weight