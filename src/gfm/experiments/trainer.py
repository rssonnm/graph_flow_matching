import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# Professional Import Structure
from ..models.vector_field import EquivariantVectorField
from ..core.optimal_transport import OptimalTransportConditionalFlowMatching
from ..core.diffusion import GeometricDiffusion
from ..core.rectified_flow import RectifiedFlow
from ..core.integration import ODESolver
from ..data.loader import GraphData

class GenerativeTrainer:
    """
    Trainer for Geometric Flow Matching models.
    Manages the training loop, optimization, and evaluation/sampling.
    """
    def __init__(self, 
                 data: GraphData, 
                 hidden_dim: int = 64, 
                 lr: float = 1e-3, 
                 device: str = None,
                 algorithm: str = 'ot-cfm'):
        """
        Args:
            algorithm: 'ot-cfm', 'diffusion', or 'rectified-flow'
        """
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device} | Algorithm: {algorithm}")
        self.algorithm = algorithm
        self.data = data.to(self.device)
        self.dim = data.x.shape[1]
        
        # Initialize SE(3)-Equivariant Model
        self.net = EquivariantVectorField(in_dim=self.dim, 
                                          hidden_dim=hidden_dim, 
                                          out_dim=self.dim, 
                                          use_atom_types=True).to(self.device)
                                          
        # Initialize Algorithm
        if algorithm == 'ot-cfm':
            self.solver_flow = OptimalTransportConditionalFlowMatching(sigma=0.0)
        elif algorithm == 'rectified-flow':
            self.solver_flow = RectifiedFlow(sigma=0.0)
        elif algorithm == 'diffusion':
            self.solver_flow = GeometricDiffusion()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
    def train_step(self) -> float:
        self.net.train()
        self.optimizer.zero_grad()
        
        # Data Batching (Currently single-batch overfitting for demonstration)
        x1 = self.data.x.unsqueeze(0) 
        adj = self.data.adj.unsqueeze(0)
        atom_types = self.data.atom_types.unsqueeze(0)
        
        # Source noise distribution N(0, I)
        x0 = torch.randn_like(x1)
        
        # Compute Loss
        loss = self.solver_flow.compute_loss(self.net, x1, x0, adj, atom_types=atom_types)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def fit(self, n_steps: int = 1000, log_interval: int = 100, save_dir: str = 'checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Initializing training on {self.device}...")
        
        for step in range(n_steps):
            loss = self.train_step()
            
            if (step + 1) % log_interval == 0:
                print(f"[Step {step+1}] Loss: {loss:.6f}")
                
        torch.save(self.net.state_dict(), os.path.join(save_dir, "model_final.pt"))
        import time
        print(f"Training successfully completed.")

    def sample(self, n_samples: int = 1, save_path: str = "trajectory.xyz"):
        self.net.eval()
        
        # Different sampling method for Diffusion vs Flow
        if self.algorithm == 'diffusion':
            # Diffusion SDE Solver (Reverse Process)
            # This is a placeholder as full SDE solver needs T steps scaling.
            # Ideally we use the same ODE solver but with probability flow ODE parameterization.
            # For VP-SDE, ODE is: dx = [ -0.5*beta(t)*x - 0.5*beta(t)*score ] dt
            # Our ODESolver solves v = dx/dt.
            # So we need to wrap the diffusion model score output into a drift term.
            print("Warning: Diffusion sampling implementation requires Score-to-Drift wrapping. Using naive ODE.")
            solver = ODESolver(self.net, method='rk4')
        else:
            solver = ODESolver(self.net, method='rk4')
        
        x0 = torch.randn(n_samples, self.data.num_nodes, self.dim).to(self.device)
        adj = self.data.adj.unsqueeze(0).repeat(n_samples, 1, 1).to(self.device)
        atom_types = self.data.atom_types.unsqueeze(0).repeat(n_samples, 1).to(self.device)
        
        
        # Integrate ODE
        x_final, trajectory = solver.solve(x0, adj, atom_types=atom_types)
        
        # Unscale trajectory for meaningful visualization
        if hasattr(self.data, 'norm_scale'):
            scale = self.data.norm_scale.to(self.device)
            trajectory = [x * scale for x in trajectory]
        
        self._save_xyz(trajectory, self.data.atom_types, save_path)
        print(f"Generated samples saved to {save_path}")

    def _save_xyz(self, trajectory: list, atom_types: torch.Tensor, filepath: str):
        pt_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
        
        with open(filepath, 'w') as f:
            for t_idx, x in enumerate(trajectory):
                # Take first sample in batch for visualization
                x_np = x[0].detach().cpu().numpy()
                n_atoms = x_np.shape[0]
                f.write(f"{n_atoms}\n")
                f.write(f"Frame {t_idx} - Generated by GFM\n")
                for i in range(n_atoms):
                    atom_num = atom_types[i].item()
                    symbol = pt_map.get(atom_num, 'X')
                    f.write(f"{symbol} {x_np[i, 0]:.6f} {x_np[i, 1]:.6f} {x_np[i, 2]:.6f}\n")
