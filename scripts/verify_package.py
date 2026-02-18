import sys
import os
sys.path.append(os.path.abspath('src'))

from gfm.data.loader import smile_to_graph
from gfm.experiments.trainer import GenerativeTrainer
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Target: Caffeine
    try:
        data = smile_to_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    except ImportError:
        print("RDKit required.")
        return

    print(f"Molecule Nodes: {data.num_nodes}")
    
    trainer = GenerativeTrainer(data, hidden_dim=64, device=device)
    trainer.fit(n_steps=500, log_interval=100) # Short run for verification
    
    os.makedirs('results', exist_ok=True)
    trainer.sample(save_path='results/test_gfm.xyz')

if __name__ == "__main__":
    main()
