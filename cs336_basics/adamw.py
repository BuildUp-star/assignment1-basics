import torch
import torch.nn
from torch.optim.optimizer import Optimizer
import math

class AdamW(Optimizer):
    r"""
    AdamW optimizer: Adam with decoupled Weight Decay (L2) as in Loshchilov & Hutter (2019).

    Hyper-parameters:
        lr (alpha): base learning rate
        betas: tuple (beta1, beta2) for first/second moment EMA
        eps (epsilon): numerical stability term added to denominator
        weight_decay (lambda): decoupled weight decay coefficient

    This class follows the pseudocode:

        init(theta); m <- 0; v <- 0
        for t = 1..T:
            g <- grad
            m <- beta1*m + (1-beta1)*g
            v <- beta2*v + (1-beta2)*g^2
            alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)   # bias correction
            theta <- theta - alpha_t * m / (sqrt(v) + eps)         # Adam update
            theta <- theta - alpha * lambda * theta                # decoupled weight decay

    Notes on implementation details:
    - We store per-parameter state in self.state[p]: {'step', 'exp_avg'(m), 'exp_avg_sq'(v)}.
    - We reject sparse gradients (AdamW in this form does not support them).
    - We use in-place ops for performance and to match common optimizer style.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()  # ensure parameter updates are not tracked by autograd
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Where does g (gradient) come from?
        -------------------------------------------------------
        After you call loss.backward(), PyTorch writes the gradient of each
        parameter p into p.grad. So here we read:
            g = p.grad (or p.grad.data in older styles)
        -------------------------------------------------------
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']                # alpha in the pseudocode
            beta1, beta2 = group['betas']
            eps = group['eps']
            lam = group['weight_decay']     # lambda (weight decay)

            for p in group['params']:
                if p.grad is None:
                    # No gradient computed for this parameter in this step
                    continue

                grad = p.grad
                # Safety check: AdamW here does not support sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # ----- State init (m, v, step) -----
                state = self.state[p]
                if len(state) == 0:
                    # step starts at 0; we will increment to 1 for the first update
                    state['step'] = 0
                    # m (first moment) and v (second moment) tensors with same shape as p
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']       # m
                exp_avg_sq = state['exp_avg_sq'] # v

                # ----- t <- t + 1 -----
                state['step'] += 1
                t = state['step']

                # Cast grad if needed (mixed-precision compatibility)
                if grad.dtype != p.dtype:
                    grad = grad.to(p.dtype)

                # ----- m <- beta1*m + (1-beta1)*g -----
                # in-place: exp_avg = beta1*exp_avg + (1-beta1)*grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # ----- v <- beta2*v + (1-beta2)*g^2 -----
                # in-place: exp_avg_sq = beta2*exp_avg_sq + (1-beta2)*(grad*grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # ----- alpha_t (bias correction) -----
                # Correct the initialization bias of EMAs:
                #   m_hat = exp_avg / (1 - beta1^t)
                #   v_hat = exp_avg_sq / (1 - beta2^t)
                # Combine with lr to avoid extra tensors:
                #   alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # ----- θ <- θ - alpha_t * m / (sqrt(v) + eps) -----
                # denom = sqrt(v_hat) + eps, but we fold bias correction into alpha_t
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

                # ----- Decoupled Weight Decay: θ <- θ - lr * lambda * θ -----
                # This is NOT L2 in the loss; it is a direct shrink on parameters.
                if lam != 0.0:
                    p.add_(p, alpha=-lr * lam)

        return loss