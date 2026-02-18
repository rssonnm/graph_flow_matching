import torch

class ODESolver:
    def __init__(self, net, method='euler'):
        self.net = net
        self.method = method

    @torch.no_grad()
    def solve(self, x0, adj, t_span=None, n_steps=100, **kwargs):
        """
        Solves the ODE dx/dt = v(x, t) from t=0 to t=1.
        x0: Initial condition (usually noise from N(0, I) or source distribution).
        t_span: Tuple (t_start, t_end). Defaults to (0, 1).
        kwargs: Additional arguments for the network (e.g. atom_types).
        """
        if t_span is None:
            t_span = (0, 1)
        
        t0, t1 = t_span
        device = x0.device
        
        # Time steps
        t_eval = torch.linspace(t0, t1, n_steps + 1, device=device)
        dt = (t1 - t0) / n_steps
        
        x = x0
        trajectory = [x.clone()]
        
        for i in range(n_steps):
            t = t_eval[i]
            # Time tensor for network: [B]
            # Assuming x0 is [B, N, D], we need t to be [B]
            B = x0.shape[0]
            t_batch = torch.full((B,), t.item(), device=device)
            
            if self.method == 'euler':
                v = self.net(x, t_batch, adj, atom_types=kwargs.get('atom_types'))
                x = x + v * dt
                
            elif self.method == 'rk4':
                # k1
                k1 = self.net(x, t_batch, adj, atom_types=kwargs.get('atom_types'))
                
                # k2
                t_mid = t + 0.5 * dt
                t_mid_batch = torch.full((B,), t_mid.item(), device=device)
                k2 = self.net(x + 0.5 * dt * k1, t_mid_batch, adj, atom_types=kwargs.get('atom_types'))
                
                # k3
                k3 = self.net(x + 0.5 * dt * k2, t_mid_batch, adj, atom_types=kwargs.get('atom_types'))
                
                # k4
                t_next = t + dt
                t_next_batch = torch.full((B,), t_next.item(), device=device)
                k4 = self.net(x + dt * k3, t_next_batch, adj, atom_types=kwargs.get('atom_types'))
                
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            trajectory.append(x.clone())
            
        return x, trajectory
