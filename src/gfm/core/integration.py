import torch
from typing import Tuple, List, Optional
import torch.nn as nn

class ODESolver:
    """
    Numerical Integrator for Ordinary Differential Equations.
    Solves dx/dt = v(x, t).
    """
    def __init__(self, net: nn.Module, method: str = 'rk4'):
        self.net = net
        self.method = method

    @torch.no_grad()
    def solve(self, 
              x0: torch.Tensor, 
              adj: torch.Tensor, 
              t_span: Tuple[float, float] = (0.0, 1.0), 
              n_steps: int = 100, 
              **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Integrates the trajectory from t_start to t_end.

        Args:
            x0: Initial state [B, N, D]
            adj: Adjacency matrix or graph structure
            t_span: (t_start, t_end)
            n_steps: Number of integration steps
            **kwargs: Additional args for the network

        Returns:
            x_final: State at t_end
            trajectory: List of states at each step
        """
        t0, t1 = t_span
        device = x0.device
        
        # Time discretization
        t_eval = torch.linspace(t0, t1, n_steps + 1, device=device)
        dt = (t1 - t0) / n_steps
        
        x = x0
        trajectory = [x.clone()]
        
        B = x0.shape[0]
        
        for i in range(n_steps):
            t = t_eval[i]
            
            # Prepare batch time for network
            t_batch = torch.full((B,), t.item(), device=device)
            
            if self.method == 'euler':
                v = self.net(x, t_batch, adj, **kwargs)
                x = x + v * dt
                
            elif self.method == 'rk4':
                # Classical Runge-Kutta 4th Order
                
                # k1
                v1 = self.net(x, t_batch, adj, **kwargs)
                k1 = v1
                
                # k2
                t_mid = t + 0.5 * dt
                t_mid_batch = torch.full((B,), t_mid.item(), device=device)
                v2 = self.net(x + 0.5 * dt * k1, t_mid_batch, adj, **kwargs)
                k2 = v2
                
                # k3
                v3 = self.net(x + 0.5 * dt * k2, t_mid_batch, adj, **kwargs)
                k3 = v3
                
                # k4
                t_next = t + dt
                t_next_batch = torch.full((B,), t_next.item(), device=device)
                v4 = self.net(x + dt * k3, t_next_batch, adj, **kwargs)
                k4 = v4
                
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            trajectory.append(x.clone())
            
        return x, trajectory
