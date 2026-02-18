import torch
import torch.nn as nn
import numpy as np

class GeometricDiffusion:
    """
    Variance Preserving Diffusion Model (VP-SDE) on Geometric Graphs.
    Used as a baseline to compare against Flow Matching.
    
    Forward Process: dX_t = -0.5 * beta(t) * X_t * dt + sqrt(beta(t)) * dW_t
    Reverse Process needs score: model predicts noise epsilon.
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_alpha_bar(self, t):
        # Integral of beta(s) ds from 0 to t
        log_alpha_bar = - (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        return torch.exp(log_alpha_bar)

    def sample_q(self, x0, t, noise=None):
        """
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        alpha_bar = self.get_alpha_bar(t).view(-1, 1, 1)
        mean = torch.sqrt(alpha_bar) * x0
        std = torch.sqrt(1 - alpha_bar)
        
        x_t = mean + std * noise
        return x_t, noise

    def compute_loss(self, net, x1, x0, adj, atom_types=None, **kwargs):
        """
        Standardized Interface:
        x1: Data (Target)
        x0: Noise (Source) - In diffusion, we diffuse x1 -> x_t. x0 argument is ignored or used as noise container.
        """
        # In VP-SDE, we start from Data (x1) and add noise.
        # Flow Matching notation: x1=Data, x0=Noise.
        # Diffusion notation: x0=Data.
        # So here: local_x0 = x1 (Data passed from trainer)
        
        real_data = x1
        B, N, D = real_data.shape
        device = real_data.device
        
        t = torch.rand(B, device=device)
        noise = torch.randn_like(real_data)
        
        # Diffuse data to x_t
        x_t, target_noise = self.sample_q(real_data, t, noise)
        
        # Network predicts eps(x_t, t)
        predicted_noise = net(x_t, t, adj, atom_types=atom_types, **kwargs)
        
        # Simple MSE on noise (epsilon-matching)
        loss = torch.mean((predicted_noise - target_noise)**2)
        return loss
