import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time

# Ensure package path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gfm.data.loader import smile_to_graph
from gfm.experiments.trainer import GenerativeTrainer
from gfm.core.integration import ODESolver

# Set style
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "figure.dpi": 300
    })
except:
    pass

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
    p1 = np.dot(p1, R)
    return np.sqrt(np.mean(np.sum((p1 - p2)**2, axis=1)))

def plot_comparison(results, output_dir):
    # 1. Training Curves (Loss)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, res in results.items():
        losses = res['losses']
        window = 20
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=f"{algo}")
        else:
            ax.plot(losses, label=f"{algo}")
            
    ax.set_title("Training Loss Convergence (Normalized)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "benchmark_loss.png"))
    
    # 2. Validation Curves (RMSD)
    fig, ax = plt.subplots(figsize=(8, 5))
    has_rmsd = False
    for algo, res in results.items():
        if 'rmsds' in res and len(res['rmsds']) > 0:
            rmsds = res['rmsds']
            # Filter None
            valid_rmsds = [(i, r) for i, r in enumerate(rmsds) if r is not None]
            if valid_rmsds:
                has_rmsd = True
                ys = [r for i, r in valid_rmsds]
                # x-axis scaled to steps
                xs = np.linspace(0, len(res['losses']), len(rmsds))
                ax.plot(xs, rmsds, label=f"{algo}", marker='o', markersize=4)
                
    if has_rmsd:
        ax.set_title("Generation Quality (RMSD) vs Training Steps")
        ax.set_xlabel("Steps")
        ax.set_ylabel("RMSD (Lower is Better)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "benchmark_rmsd.png"))

def run_benchmark(args):
    device = torch.device(args.device)
    print(f"--- Benchmarking on {device} ---")
    
    data = smile_to_graph(args.smile)
    # Get GT for RMSD
    x_gt = data.x.detach().cpu().numpy()
    if hasattr(data, 'norm_scale'):
        x_gt = x_gt * data.norm_scale.item()

    results = {}
    algorithms = ['ot-cfm', 'diffusion', 'rectified-flow']
    
    for algo in algorithms:
        print(f"\nTraining Algorithm: {algo}")
        trainer = GenerativeTrainer(
            data=data,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            device=args.device,
            algorithm=algo
        )
        
        start_time = time.time()
        losses = []
        rmsds = []
        
        for step in range(args.steps):
            loss = trainer.train_step()
            losses.append(loss)
            
            # Evaluate RMSD every 20 steps
            if (step+1) % 20 == 0:
                print(f"Step {step+1}/{args.steps} | Loss: {loss:.4f}")
                
                # Fast Sample (1 sample)
                trainer.net.eval()
                with torch.no_grad():
                    # For benchmarking, we use a simple solver call
                    solver = ODESolver(trainer.net, method='euler') # Euler for speed in benchmark
                    x0 = torch.randn(1, data.num_nodes, data.dim).to(device)
                    adj = data.adj.unsqueeze(0).to(device)
                    at = data.atom_types.unsqueeze(0).to(device)
                    
                    try:
                        # Diffusion specific logic handled in trainer normally, 
                        # but for raw benchmark we try unified solver or skip
                        if algo == 'diffusion':
                             # Diffusion requires specific sampling, might be unstable with Euler ODE
                             # Use RK4 for diffusion
                             solver = ODESolver(trainer.net, method='rk4')
                        
                        x_final, _ = solver.solve(x0, adj, atom_types=at)
                        sample = x_final[0].cpu().numpy()
                        
                        # Unscale
                        if hasattr(data, 'norm_scale'):
                            sample = sample * data.norm_scale.item()
                            
                        rmsd = compute_rmsd(sample, x_gt)
                        rmsds.append(rmsd)
                    except Exception as e:
                        print(f"Sampling failed: {e}")
                        rmsds.append(None)
                
                trainer.net.train()

        
        duration = time.time() - start_time
        # Final avg loss
        final_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
        
        results[algo] = {
            'losses': losses,
            'rmsds': rmsds,
            'duration': duration,
            'final_loss': final_loss
        }
        
        # Save Model
        algo_dir = os.path.join(args.output_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)
        torch.save(trainer.net.state_dict(), os.path.join(algo_dir, 'model.pt'))
        
    # Plot Comparison
    plot_comparison(results, args.output_dir)
    print(f"Benchmark complete. Results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smile', type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default="results/benchmark")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Auto-detect MPS
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
            
    run_benchmark(args)

if __name__ == "__main__":
    main()
