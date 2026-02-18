import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for time conditioning, following the Transformer methodology.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        encodes time t into a fixed-size vector.
        Args:
            time: [B] tensor of time steps.
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # [B, 1] * [1, half_dim] -> [B, half_dim]
        embeddings = time[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
