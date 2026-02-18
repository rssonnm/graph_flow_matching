import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_graph_snapshot(x, adj, title, save_path=None):
    """
    Plots a 2D graph snapshot.
    x: [N, 2]
    adj: [N, N]
    """
    x_np = x.detach().cpu().numpy()
    adj_np = adj.detach().cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    
    # Plot edges
    rows, cols = np.where(adj_np > 0)
    for r, c in zip(rows, cols):
        if r < c: # unique edges
            plt.plot([x_np[r, 0], x_np[c, 0]], [x_np[r, 1], x_np[c, 1]], 
                     color='gray', alpha=0.3, linewidth=0.5)
            
    # Plot nodes
    plt.scatter(x_np[:, 0], x_np[:, 1], s=20, c='blue', alpha=0.8)
    
    plt.title(title)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_trajectories(trajectory, adj, save_dir):
    """
    Plots the evolution of the graph.
    trajectory: List of tensors [N, 2]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot selected frames
    n_frames = len(trajectory)
    indices = np.linspace(0, n_frames - 1, 6, dtype=int)
    
    for i in indices:
        x = trajectory[i]
        t_val = i / (n_frames - 1)
        plot_graph_snapshot(x, adj, f"t = {t_val:.2f}", 
                            save_path=f"{save_dir}/frame_{i:03d}.png")

def plot_vector_field(net, t, adj, device, save_path=None):
    """
    Plots the vector field v_t(x) on a grid.
    Only makes sense if we ignore the graph structure for the background field,
    but our GNN depends on adj. 
    So instead, we can plot the velocity vectors at the current node positions.
    """
    pass # Complex to visualize GNN vector field over space without defining "nodes" everywhere. 
         # We will stick to trajectory visualization.
