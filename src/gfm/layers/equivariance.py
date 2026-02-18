import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Implements the equivariant message passing scheme described in:
    Satorras, V. G., et al. (2021). "E(n) Equivariant Graph Neural Networks". ICML.
    
    Updates node coordinates x and features h such that the transformation is SE(3)-equivariant.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 edge_attr_dim: int = 0, 
                 residual: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_attr_dim = edge_attr_dim
        self.residual = residual

        # Edge Model: phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        # Input: 2 * input_dim (h_i, h_j) + 1 (squared distance) + edge_attr_dim
        edge_input_dim = input_dim * 2 + 1 + edge_attr_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node Model: phi_h(h_i, sum(m_ij))
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Coordinate Model: phi_x(m_ij)
        # Updates x_i += C * sum((x_i - x_j) * phi_x(m_ij))
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False) 
        )
        
        # Initialize last layer for stability near zero updates initially
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(self, 
                h: torch.Tensor, 
                x: torch.Tensor, 
                adj: Optional[torch.Tensor] = None, 
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for EGNN Layer.

        Args:
            h: Node features [B, N, D_h]
            x: Node coordinates [B, N, 3]
            adj: Adjacency matrix [B, N, N] (Optional mask)
            edge_attr: Edge attributes [B, N, N, D_e] (Optional)

        Returns:
            h_new: Updated node features
            x_new: Updated node coordinates
        """
        B, N, _ = h.shape

        # 1. Compute Relative Squared Distances
        # x_diff = x_i - x_j
        x_diff = x.unsqueeze(2) - x.unsqueeze(1) # [B, N, N, 3]
        d_sq = torch.sum(x_diff ** 2, dim=-1, keepdim=True) # [B, N, N, 1]
        
        # 2. Construct Edge Inputs
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        
        edge_inputs = [h_i, h_j, d_sq]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
            
        edge_inputs = torch.cat(edge_inputs, dim=-1) # [B, N, N, dims]
        
        # 3. Compute Edge Messages m_ij
        m_ij = self.edge_mlp(edge_inputs) # [B, N, N, hidden_dim]
        
        # Apply Adjacency Masking if provided
        if adj is not None:
             mask = adj.unsqueeze(-1)
             m_ij = m_ij * mask
             x_diff = x_diff * mask # Also mask coordinate updates from unconnected nodes

        # 4. Coordinate Update (Equivariant)
        # Trans: phi_x(m_ij)
        trans = self.coord_mlp(m_ij) # [B, N, N, 1]
        
        # Aggregation: sum_j (x_i - x_j) * trans_ij
        # Note: x_diff is defined as x_i - x_j locally? 
        # Above: x.unsqueeze(2) - x.unsqueeze(1) -> x[i] - x[j]
        # Standard EGNN update is x_i += sum (x_i - x_j) * phi_x
        # which pushes particles apart or together along the bond vector.
        x_agg = torch.sum(x_diff * trans, dim=2) # [B, N, 3]
        x_new = x + x_agg
        
        # 5. Node Feature Update (Invariant)
        # Aggregation: sum_j m_ij
        m_i = torch.sum(m_ij, dim=2) # [B, N, hidden_dim]
        
        node_inputs = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(node_inputs)
        
        # Residual Connection
        if self.residual and h.shape[-1] == h_new.shape[-1]:
            h_new = h + h_new
            
        return h_new, x_new
