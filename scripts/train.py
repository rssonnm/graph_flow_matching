import argparse
import torch
import sys
import os
from gfm.data.loader import smile_to_graph
from gfm.experiments.trainer import GenerativeTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Geometric Flow Matching Model")
    parser.add_argument('--smile', type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", help="SMILES string of target molecule")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension of EGNN")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--steps', type=int, default=5000, help="Number of training steps")
    parser.add_argument('--log_interval', type=int, default=100, help="Logging interval")
    parser.add_argument('--algorithm', type=str, default="ot-cfm", choices=['ot-cfm', 'diffusion', 'rectified-flow'], help="Algorithm to use")
    parser.add_argument('--save_dir', type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument('--device', type=str, default=None, help="Device to use (cuda/mps/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"=== Geometric Flow Matching Training ===")
    print(f"Algorithm: {args.algorithm}")
    print(f"Target: {args.smile}")
    
    try:
        data = smile_to_graph(args.smile)
    except Exception as e:
        print(f"Error loading molecule: {e}")
        return

    print(f"Graph: {data.num_nodes} nodes, {data.dim} coords")
    
    # Check device
    device = args.device
    if device is None:
        if torch.backends.mps.is_available(): device = 'mps'
        elif torch.cuda.is_available(): device = 'cuda'
        else: device = 'cpu'
    
    trainer = GenerativeTrainer(
        data=data,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
        algorithm=args.algorithm
    )
    
    trainer.fit(
        n_steps=args.steps, 
        log_interval=args.log_interval,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
