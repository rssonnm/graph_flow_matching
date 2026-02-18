import torch
import torch.optim as optim
import os
from .model import VectorField
from .flow import ConditionalFlowMatching
from .solver import ODESolver
from .visualize import plot_trajectories, plot_graph_snapshot

class FlowMatchingTrainer:
    def __init__(self, target_graph, hidden_dim=64, lr=1e-3, device='cpu'):
        self.device = device
        self.target_graph = target_graph.to(device)
        self.dim = target_graph.x.shape[1]
        
        self.net = VectorField(in_dim=self.dim, hidden_dim=hidden_dim, out_dim=self.dim).to(device)
        self.flow = ConditionalFlowMatching(sigma=0.0) # OT-CFM
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
    def train_step(self, batch_size=32):
        self.net.train()
        self.optimizer.zero_grad()
        
        # 1. Prepare batch
        # For this simple "single graph geometry" task, we are learning to transport 
        # noise to the specific geometry of the target graph.
        # We can treat each node as a sample, or the whole graph as a sample.
        # Since we want to preserve graph structure (GNN), we process the whole graph.
        # Ideally we batch multiple graphs, but since we have 1 fixed target graph structure,
        # we can just replicate it or strictly learn on this one graph (overfitting to the shape).
        # Let's start with single graph training (batch_size=1 effectively for the graph structure, 
        # but we sample different t).
        
        # x1: Target [1, N, D]
        x1 = self.target_graph.x.unsqueeze(0) 
        adj = self.target_graph.adj.unsqueeze(0)
        
        # Replicate for batch_size if needed, but standard CFM usually batches over data points.
        # Here our "data point" is the whole graph configuration.
        # Let's say we want to generate *variations* of the graph? 
        # Or just transport noise to THIS graph?
        # "Graph Flow Matching" usually implies generative modeling of graph features.
        # So we want to map N(0, I) -> P(data).
        # If we only have 1 data graph, we map to a delta distribution.
        # To make it interesting, let's assume we have a distribution of graphs (e.g. rotated grids).
        # But for 'from scratch' simplicity, mapping N(0,1) to the single Grid Graph is fine.
        
        # x0: Source ~ N(0, 1) [1, N, D]
        x0 = torch.randn_like(x1)
        
        loss = self.flow.compute_loss(self.net, x1, x0, adj)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, n_steps=1000, log_interval=100, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training on {self.device}...")
        for step in range(n_steps):
            loss = self.train_step()
            
            if (step + 1) % log_interval == 0:
                print(f"Step {step+1}: Loss = {loss:.6f}")
                
        torch.save(self.net.state_dict(), f"{save_dir}/model.pt")
        print("Training complete.")

    def sample_and_visualize(self, save_dir='results'):
        self.net.eval()
        solver = ODESolver(self.net, method='rk4')
        
        # Sample x0
        x0 = torch.randn(1, self.target_graph.num_nodes, self.dim).to(self.device)
        adj = self.target_graph.adj.unsqueeze(0).to(self.device)
        
        # Solve ODE
        x_final, trajectory = solver.solve(x0, adj)
        
        # Visualize
        # Remove batch dim for plotting
        traj_unbatched = [x[0] for x in trajectory]
        
        plot_trajectories(traj_unbatched, self.target_graph.adj, save_dir)
        print(f"Visualization saved to {save_dir}")
