from collections.abc import Callable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt

# --- Custom SGD optimizer with learning-rate decay ---
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}  # Default hyperparameters for all param groups
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform one optimization step.
        If closure is provided, it will be called to re-evaluate the model and loss.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Learning rate for this group
            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip parameters without gradients
                state = self.state[p]      # Get state dictionary for parameter p
                t = state.get("t", 0)      # Step count for this parameter
                grad = p.grad.data         # Gradient tensor
                # Parameter update with decay: lr_t = lr / sqrt(t+1)
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1         # Increment step count
        return loss

# --- Experiment function ---
def run_experiment(lrs, steps=10):
    """
    Run optimization for each learning rate in `lrs` for `steps` iterations.
    Returns a dictionary mapping lr -> list of loss values.
    """
    results = {}
    torch.manual_seed(0)  # Reproducibility
    for lr in lrs:
        # Initialize weights for each learning rate test
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []
        for _ in range(steps):
            opt.zero_grad()                 # Clear old gradients
            loss = (weights ** 2).mean()     # Forward pass: simple quadratic loss
            losses.append(loss.item())
            loss.backward()                  # Backward pass: compute gradients
            opt.step()                       # Update parameters
        results[lr] = losses
    return results

if __name__ == "__main__":
    # Learning rates to test
    lrs = [1e1, 1e2, 1e3]
    results = run_experiment(lrs, steps=10)

    # Plot loss curves
    plt.figure(figsize=(6, 4))
    for lr, losses in results.items():
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f"lr={lr:g}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step for different learning rates (SGD with decay)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print numeric results for reference
    for lr, losses in results.items():
        print(f"lr={lr:g} -> losses: {losses}")
