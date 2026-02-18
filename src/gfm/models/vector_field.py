import torch
import torch.nn as nn
from typing import Optional

from .components import SinusoidalPositionEmbeddings
# Relative imports within logical package structure
from ..layers.equivariance import EGNNLayer

class EquivariantVectorField(nn.Module):
    """
    SE(3)-Equivariant Vector Field Neural Network.
    
    Approximates the time-dependent vector field v_t(x) = dx/dt required for 
    Continuous Normalizing Flows on geometric graphs.
    
    Architecture:
        - Time Embedding (Sinusoidal)
        - Feature Embedding (Atomic Types)
        - Stacked EGNN Layers for equivariant message passing
        - Velocity Readout
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int = 4, 
                 use_atom_types: bool = False,
                 num_atom_types: int = 100):
        super().__init__()
        self.use_atom_types = use_atom_types
        
        # 1. Time Conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Input Projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        if self.use_atom_types:
            self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)

        # 3. Equivariant Layers
        self.layers = nn.ModuleList([
            EGNNLayer(input_dim=hidden_dim, 
                      hidden_dim=hidden_dim, 
                      output_dim=hidden_dim, 
                      residual=True) 
            for _ in range(num_layers)
        ])
        
        # 4. Output Projection (Optional if needed for features, but velocity comes from coordinate updates)
        # Note: In this architecture, velocity is derived from the cumulative coordinate displacement.
        
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                adj: Optional[torch.Tensor] = None, 
                atom_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the vector field v(x, t).

        Args:
            x: Node Coordinates [B, N, 3]
            t: Time scalar [B]
            adj: Adjacency matrix [B, N, N]
            atom_types: Atomic numbers [B, N]

        Returns:
            v: Velocity vector field [B, N, 3]
        """
        B, N, _ = x.shape
        
        # Embed Time
        t_emb = self.time_mlp(t) # [B, H]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1) # [B, N, H]
        
        # Embed Features / Atoms
        if self.use_atom_types and atom_types is not None:
            h = self.atom_emb(atom_types) # [B, N, H]
        else:
            # Initialize with zero features projected
            h = torch.zeros(B, N, self.input_proj.in_features, device=x.device)
            h = self.input_proj(h)

        # Inject Time Signal
        h = h + t_emb
        
        # Store initial coordinates to compute displacement
        x_in = x.clone()
        
        # Iterate through Equivariant Layers
        for layer in self.layers:
            # Re-inject time signal at each layer (skip connection strategy)
            h = h + t_emb
            h, x = layer(h, x, adj)
            
        # The velocity is the total displacement learned by the network
        # v = x_result - x_initial
        # This interpretation aligns with residual flows and simplifies training.
        v = x - x_in
        
        return v
