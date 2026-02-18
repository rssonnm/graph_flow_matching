import argparse
import torch
import os
from gfm.data.loader import smile_to_graph
from gfm.experiments.trainer import GenerativeTrainer
from gfm.models.vector_field import EquivariantVectorField

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Samples from GFM Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model.pt")
    parser.add_argument('--smile', type=str, default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", help="SMILES for topology reference")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension (must match training)")
    parser.add_argument('--output', type=str, default="results/generated.xyz", help="Output XYZ file")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"=== Geometric Flow Matching Generation ===")
    print(f"Loading model from: {args.model_path}")
    
    try:
        data = smile_to_graph(args.smile)
    except Exception as e:
        print(f"Error loading topology: {e}")
        return

    # Initialize Trainer Wrapper (simplifies loading)
    trainer = GenerativeTrainer(
        data=data,
        hidden_dim=args.hidden_dim,
        device=args.device
    )
    
    # Load Weights
    if os.path.exists(args.model_path):
        trainer.net.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model weights loaded.")
    else:
        print(f"Error: Model path {args.model_path} does not exist.")
        return
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trainer.sample(n_samples=args.n_samples, save_path=args.output)
    print(f"Generation complete.")

if __name__ == "__main__":
    main()
