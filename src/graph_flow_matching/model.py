import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GCNLayer(nn.Module):
    """
    Simple GCN Layer: H' = A H W + b
    Optionally with residual connection and normalization.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [B, N, D] or [N, D]
        # adj: [B, N, N] or [N, N]
        
        # Simple propagation: A * X
        # Note: We assume A is already normalized or we do it here. 
        # For this simple implementation, we assume A includes self-loops and is symmetrically normalized 
        # outside or we just do raw message passing A * X.
        
        # Support batching
        if x.dim() == 3:
            out = torch.bmm(adj, x)
        else:
            out = torch.mm(adj, x)
            
        out = self.linear(out)
        return out

class VectorField(nn.Module):
    """
    Time-dependent Graph Neural Network approximating the vector field v_t(x).
    Arguments:
        x: Node features/positions [B, N, D]
        t: Time [B]
        adj: Adjacency matrix [B, N, N]
        atom_types: (Optional) Atomic numbers [B, N]
    Output:
        v: Velocity [B, N, D]
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, use_atom_types=False):
        super().__init__()
        self.use_atom_types = use_atom_types
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        if self.use_atom_types:
            # Atomic numbers usually go up to 118, let's say max 100 common ones.
            self.atom_emb = nn.Embedding(100, hidden_dim)
        
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t, adj, atom_types=None):
        # Time embedding
        t_emb = self.time_mlp(t) # [B, H]
        
        # Expand time embedding to nodes: [B, H] -> [B, N, H] (if batched)
        # or [H] -> [N, H] (if unbatched)
        
        h = self.input_proj(x)
        
        if self.use_atom_types and atom_types is not None:
             a_emb = self.atom_emb(atom_types) # [B, N, H] or [N, H]
             h = h + a_emb
        
        B, N, _ = h.shape if h.dim() == 3 else (1, h.shape[0], h.shape[1])
        
        if t_emb.dim() == 2 and h.dim() == 3:
             t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        elif t_emb.dim() == 2 and h.dim() == 2:
             # Should generally be batched, but handling unbatched case just in case
             pass

        for layer, norm in zip(self.layers, self.norms):
            h_in = h
            # Message Passing
            h = layer(h, adj)
            
            # Add time info
            h = h + t_emb
            
            # Non-linearity & Norm
            h = F.silu(norm(h))
            
            # Residual
            h = h + h_in
            
        out = self.final_proj(h)
        return out
