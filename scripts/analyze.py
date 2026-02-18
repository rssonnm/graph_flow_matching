import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure package is in path if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gfm.models.vector_field import EquivariantVectorField
from gfm.core.integration import ODESolver
from gfm.data.loader import smile_to_graph
from scipy.spatial.distance import pdist, squareform

# Set style
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "figure.dpi": 300
    })
except:
    pass # Fallback if style not found

def compute_rmsd(pos_1, pos_2):
    # Centering
    p1 = pos_1 - np.mean(pos_1, axis=0)
    p2 = pos_2 - np.mean(pos_2, axis=0)
    
    # Kabsch
    H = np.dot(p1.T, p2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    p1_aligned = np.dot(p1, R)
    rmsd = np.sqrt(np.mean(np.sum((p1_aligned - p2)**2, axis=1)))
    return rmsd

def analyze(args):
    device = torch.device(args.device)
    
    # Load Ground Truth
    data = smile_to_graph(args.smile)
    x_gt = data.x.numpy()
    
    # Load Model
    model = EquivariantVectorField(in_dim=3, hidden_dim=args.hidden_dim, out_dim=3, use_atom_types=True).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded.")
    else:
        print("Model file not found.")
        return

    model.eval()
    solver = ODESolver(model, method='rk4')
    
    # Generate Samples
    print(f"Generating {args.n_samples} samples for analysis...")
    rmsds = []
    
    for _ in range(args.n_samples):
        x0 = torch.randn(1, data.num_nodes, 3).to(device)
        adj = data.adj.unsqueeze(0).to(device)
        at = data.atom_types.unsqueeze(0).to(device)
        
        with torch.no_grad():
            x_final, _ = solver.solve(x0, adj, atom_types=at)
            sample = x_final[0].cpu().numpy()
            
        rmsds.append(compute_rmsd(sample, x_gt))
        
    # Plot RMSD
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    sns.histplot(rmsds, kde=True, color='teal', bins=15)
    plt.title(f"Stability Analysis (n={args.n_samples})")
    plt.xlabel("RMSD to Ground Truth (A)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "analysis_rmsd.png"))
    print(f"Analysis saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--smile', type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default="results/analysis")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    analyze(args)

if __name__ == "__main__":
    main()
