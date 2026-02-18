import torch
import torch.nn as nn

class ConditionalFlowMatching:
    def __init__(self, sigma: float = 0.0):
        """
        Args:
            sigma: Noise level for the path (usually 0 for exact OT path, or small > 0).
        """
        self.sigma = sigma

    def sample_location_and_target_velocity(self, x0, x1, t):
        """
        Computes x_t and u_t(x|x1).
        x_t = (1 - (1 - sigma) * t) * x0 + t * x1  (if we consider general case, but simplified below)
        
        Standard OT-CFM path:
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0
        """
        # t is [B] or [B, 1]. Ensure broadcasting.
        if t.dim() == 1:
            t = t.view(-1, 1, 1) # [B, 1, 1] for [B, N, D]
        
        # Linear interpolation
        t_expand = t
        
        # x_t
        mu_t = (1 - t_expand) * x0 + t_expand * x1
        x_t = mu_t + self.sigma * torch.randn_like(x0) # Add noise if sigma > 0 (not used in standard OT-CFM typically for the mean, but for variance)
        
        # Target velocity u_t(x|x1) = x1 - x0 for deterministic path
        # If sigma > 0, it's slightly different, but let's stick to deterministic OT path for now.
        u_t = x1 - x0 
        
        return x_t, u_t

    def compute_loss(self, net, x1, x0, adj, **kwargs):
        """
        Computes the CFM loss.
        """
        B, N, D = x1.shape
        device = x1.device
        
        # 1. Sample t uniformly
        t = torch.rand(B, device=device)
        
        # 2. Sample x_t and target velocity u_t
        x_t, u_t = self.sample_location_and_target_velocity(x0, x1, t)
        
        # 3. Predict velocity
        # net needs t to be [B] usually? Let's check model.py
        # model.py: embeddings = time[:, None] * embeddings[None, :] -> implies t is [B] 1D tensor
        v_t = net(x_t, t, adj, **kwargs)
        
        # 4. MSE Loss
        loss = torch.mean((v_t - u_t)**2)
        
        return loss
