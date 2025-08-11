from typing import Optional
import torch
import torch.nn as nn
from torch import device, dtype
from cs336_basics.linear import Linear
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
import math


class SwiGLU(nn.Module):
    """
    A position-wise feed-forward network using the SwiGLU activation.
    This implementation follows the Llama 2 architecture for its FFN layers.
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, device: device | None = None, dtype: dtype | None = None):
        super().__init__()

        # 1. Calculate the intermediate dimension (d_ff) as per the instructions.
        # This is a common practice in modern LLMs like Llama.
        if d_ff is None:
            hidden_dim_approx = int(d_model * 8 / 3)
            # Ensure d_ff is a multiple of 64 for hardware efficiency.
            # This is an efficient way to round up to the nearest multiple.
            d_ff = ((hidden_dim_approx + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
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

#Can't use this because of numerical stability issues with large logits.
'''
def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    sm = softmax(inputs, dim=-1)  # Apply softmax to the last dimension (vocab_size)
    probs = sm.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(1)  # Gather probabilities for the target indices
    log_probs = probs.clamp_min(1e-12).log()  # Take the log of the probabilities
    loss = -log_probs.mean()  # Compute the mean negative log probability
    return loss
'''

def cross_entropy(
    inputs: Float[Tensor, "... vocab_size"],   # logits (NOT probabilities)
    targets: Int[Tensor, "..."],               # class indices with shape == inputs.shape[:-1]
) -> Float[Tensor, ""]:
    """
    Numerically-stable cross-entropy computed directly from logits, without torch.logsumexp.
    Handles arbitrary batch dimensions; the last dimension is vocab_size.
    """
    # Ensure dtypes are right (targets must be long for gather)
    if targets.dtype != torch.long:
        targets = targets.long()

    # 1) subtract max for numerical stability
    # max_vals: [B, 1]
    max_vals = inputs.max(dim=1, keepdim=True).values
    centered = inputs - max_vals              # [B, V]

    # 2) exp and sum over classes
    sum_exp = centered.exp().sum(dim=1)       # [B]

    # 3) manual logsumexp = max + log(sum(exp(centered)))
    lse = max_vals.squeeze(1) + sum_exp.log() # [B]

    # 4) pick the target class logit (original logits OK; or use centered + max equivalently)
    y_logit = inputs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [B]

    # 5) CE per-sample and average
    loss = (lse - y_logit).mean()
    return loss


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
        # res[mask == False] = float("-inf")  # Set masked positions to -inf for softmax
        res = res.masked_fill(mask, -torch.inf)
    res = softmax(res, dim=-1)  # Apply softmax to the last dimension
    res = res @ V  # Multiply by the values tensor
    return res

#uv run pytest -k test_get_lr_cosine_schedule
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Linear warmup phase
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        # Cosine decay phase
        it -= warmup_iters  # Adjust iteration count to start from 0 after warmup
        it = it / (cosine_cycle_iters - warmup_iters)  # Normalize to [0,1] for a single cosine cycle
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(it * math.pi))
    else:
        # After all warmup and cosine iterations, return min learning rate
        return min_learning_rate

from collections.abc import Iterable
#uv run pytest -k test_gradient_clipping
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by the *global* L2 norm (a.k.a. global-norm clipping).

    Algorithm (global norm):
      1) Conceptually concatenate all parameter gradients into a big vector g.
      2) Compute total_norm = ||g||_2 = sqrt( sum_i ||grad_i||_2^2 ).
      3) If total_norm <= M: do nothing.
      4) Else scale every gradient in-place by: scale = M / (total_norm + eps).
         This preserves each gradient's *direction* and only shrinks its *magnitude*.

    Notes:
      - Use `with torch.no_grad():` for in-place updates instead of `.data` (safer w.r.t autograd).
      - When accumulating the norm, cast to float32 for numerical stability in mixed precision.
      - Handle sparse gradients by operating on their non-zero values only:
          g.coalesce().values() for reading; coalesce + _values().mul_() for in-place scaling.
    """
    eps = 1e-6  # small constant for numerical stability

    # Materialize only parameters that *actually* have gradients
    params = [p for p in parameters if (p is not None and p.grad is not None)]
    if not params:
        return

    # Accumulate sum of squares on the grads' device, in float32 for stability
    device = params[0].grad.device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)

    # 1) Sum of squares across *all* gradients (dense + sparse)
    for p in params:
        g = p.grad
        if g.is_sparse:
            # Sparse gradients (COO): work on non-zero values only.
            # coalesce() merges duplicate indices so each coordinate is unique.
            v = g.coalesce().values().detach().float()
            total_sq += (v * v).sum()
        else:
            # Dense gradients: read-only view, cast to float32 before squaring.
            v = g.detach().float()
            total_sq += (v * v).sum()

    total_norm = total_sq.sqrt()

    # 2) If already within the limit (or literally zero), nothing to do
    if total_norm <= max_l2_norm or total_norm == 0:
        return

    # 3) Global scale factor; 0 < scale < 1 when clipping is needed
    scale = max_l2_norm / (total_norm + eps)

    # 4) In-place scaling under no-grad context (safe alternative to `.data`)
    with torch.no_grad():
        for p in params:
            g = p.grad
            if g.is_sparse:
                # IMPORTANT: scale the *values* of the sparse COO grad
                sg = g.coalesce()            # still a sparse tensor (indices + values)
                sg._values().mul_(scale)     # in-place on non-zero values (dense view)
                p.grad = sg                  # write back the *sparse* tensor (keep layout)
                # DON'T do: p.grad = g.coalesce()._values().mul_(scale)
                # That returns a dense values tensor and drops indices/layout.
            else:
                g.mul_(scale)                # dense: simple in-place multiply




