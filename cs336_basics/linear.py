import math
import torch
from torch import Tensor, device, dtype
import torch.nn as nn

class Linear(nn.Module):
    """
    A pure‐PyTorch implementation of a bias‐free linear layer:
        y = W x

    Follows the interface of torch.nn.Linear, except:
      - no bias
      - no `bias` argument
      - uses truncated normal init as specified
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        """
        Args:
            in_features:  size of each input sample
            out_features: size of each output sample
            device:       where to place the weight tensor (CPU/CUDA)
            dtype:        data type of the weight tensor
        """
        # 1. initialize base class
        super().__init__()

        # 2. save dims (optional, but mirrors nn.Linear)
        self.in_features = in_features
        self.out_features = out_features

        # 3. create weight parameter of shape (out_features, in_features)
        #    use factory_kwargs so device/dtype are respected
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        # 4. compute std for truncated normal: var = 2/(in+out)
        std = math.sqrt(2.0 / (in_features + out_features))

        # 5. initialize weight with truncated normal on [-3σ, 3σ]
        #    this matches: 𝒩(0, σ²=2/(d_in+d_out)), truncated at ±3σ
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the incoming data:
            x: (..., in_features)
        Returns:
            y: (..., out_features)
        """
        # x.matmul(self.weight.T) handles arbitrary leading dims
        return x.matmul(self.weight.t())
