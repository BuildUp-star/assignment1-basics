import torch
from torch import nn

class Embedding(nn.Module):
    """
    Custom Embedding module that mimics torch.nn.Embedding functionality
    without actually using nn.Embedding or nn.functional.embedding.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
        Initialize the embedding matrix.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of each embedding vector.
            device (torch.device, optional): Device for parameters (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype, optional): Data type for parameters (e.g., torch.float32).
        """
        super().__init__()
        # Store sizes for reference
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding weight parameter of shape (num_embeddings, embedding_dim)
        # torch.empty respects device and dtype arguments
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        # Initialize weights using truncated normal distribution.
        # Here, we use a standard deviation of 1/sqrt(embedding_dim) to scale appropriately.
        # You can adjust mean and std based on model requirements.
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=embedding_dim ** -0.5
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform embedding lookup.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs with dtype torch.long,
                                      shape (batch_size, sequence_length) or any shape.

        Returns:
            torch.Tensor: Corresponding embeddings of shape
                          (*token_ids.shape, embedding_dim).
        """
        # Ensure token_ids is of integer type for indexing
        if not token_ids.dtype == torch.long:
            token_ids = token_ids.long()

        # Use tensor indexing to gather rows from self.weight
        # This returns a new tensor with an extra trailing dimension for embeddings
        embeddings = self.weight[token_ids]

        return embeddings
