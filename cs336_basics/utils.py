import torch
import torch.nn as nn
from torch import device, dtype
from cs336_basics.linear import Linear

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