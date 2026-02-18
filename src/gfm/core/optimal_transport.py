import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Tuple

class OptimalTransportConditionalFlowMatching:
    """
    Implements Optimal Transport Conditional Flow Matching (OT-CFM).
    
    References:
    Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling". ICLR.
    Tong, A., et al. (2023). "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport".
    
    Attributes:
        sigma (float): Noise scale for the path construction (typically 0.0 for deterministic straight paths).
    """
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_location_and_target_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the target conditional path x_t and velocity u_t using linear interpolation (Geodesic path).
        
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0
        
        Args:
            x0: Source sample (Noise)
            x1: Target sample (Data)
            t: Time scalar
            
        Returns:
            x_t: Interpolated state
            u_t: Target velocity
        """
        # Ensure proper broadcasting for t
        if t.dim() == 1:
            t = t.view(-1, 1, 1) # [B, 1, 1] assuming global time for batch
            
        mu_t = (1 - t) * x0 + t * x1
        
        # Add sigma-path noise if required (variance exploding/preserving paths), 
        # but standard OT-CFM uses deterministic paths relative to pairs.
        x_t = mu_t
        if self.sigma > 0:
             x_t = x_t + self.sigma * torch.randn_like(x0)
             
        # Velocity for straight line path
        u_t = x1 - x0
        
        return x_t, u_t

    def compute_loss(self, net: nn.Module, x1: torch.Tensor, x0: torch.Tensor, adj: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the Flow Matching Loss with Minibatch Optimal Transport coupling.
        
        Args:
            net: Vector field network v_theta(x, t)
            x1: Target data batch [B, N, D]
            x0: Source noise batch [B, N, D]
            adj: Adjacency matrix
            **kwargs: Additional arguments for network (e.g. atom_types)
            
        Returns:
            loss: Mean Squared Error between predicted and target velocity.
        """
        B, N, D = x1.shape
        device = x1.device
        
        # 1. Optimal Transport Coupling (Minibatch)
        # Flatten samples to vectors for distance calculation
        x0_flat = x0.view(B, -1)
        x1_flat = x1.view(B, -1)
        
        # Compute Squared Euclidean Cost Matrix
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        x0_sq = torch.sum(x0_flat**2, dim=1, keepdim=True)
        x1_sq = torch.sum(x1_flat**2, dim=1, keepdim=True)
        dist_sq = x0_sq + x1_sq.T - 2 * torch.matmul(x0_flat, x1_flat.T) # [B, B]
        
        # Solve Assignment Problem (Hungarian Algorithm)
        cost_matrix = dist_sq.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder samples to form optimal (x0, x1) pairs
        # x0[row_ind[k]] matches to x1[col_ind[k]]
        # We align x1 to x0 (or vice versa). Let's construct new tensors.
        x0_ordered = x0[row_ind]
        x1_ordered = x1[col_ind]
        
        # Also reorder auxiliary data associated with x1 (like adj, atom_types)
        if hasattr(adj, 'shape') and adj.shape[0] == B:
             adj_ordered = adj[col_ind]
        else:
             adj_ordered = adj
             
        kwargs_ordered = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                kwargs_ordered[k] = v[col_ind]
            else:
                kwargs_ordered[k] = v
        
        # 2. Stratified Sampling for t (Variance Reduction for Flow Matching)
        # Instead of random U[0,1], we divide [0,1] into B bins and sample one from each.
        # This reduces the variance of the Monte Carlo expectation over t.
        # Reference: "Flow Matching for Generative Modeling", Lipman et al. 2023
        steps = torch.arange(B, device=device) / B
        noise = torch.rand(B, device=device) / B
        t = steps + noise
        # Shuffle t to break correlation with batch index order (though data is randomized)
        t = t[torch.randperm(B, device=device)]
        
        # 3. Compute Flow Target
        x_t, u_t = self.sample_location_and_target_velocity(x0_ordered, x1_ordered, t)
        
        # 4. Predict Velocity
        v_t = net(x_t, t, adj_ordered, **kwargs_ordered)
        
        # 5. Regression Loss (Normalized)
        # The target velocity u_t = x1 - x0 has variance approx 2*Var(Data) because x1, x0 are independent (ish).
        # Diffusion predicts noise epsilon with Variance 1.
        # To make the loss magnitude comparable to Diffusion, we normalize by the expected target variance (2.0).
        # This is a mathematical scaling to align the "energy units" of the two objectives.
        
        squared_error = (v_t - u_t)**2
        loss = torch.mean(squared_error) / 2.0
        
        return loss
