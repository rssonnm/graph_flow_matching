# Geometric Flow Matching (GFM)

**A Principled Framework for SE(3)-Equivariant Generative Modeling on Geometric Graphs.**

## Overview
This repository implements a state-of-the-art framework for generative modeling of 3D molecular structures. It integrates **E(n)-Equivariant Graph Neural Networks (EGNN)** with **Optimal Transport Conditional Flow Matching (OT-CFM)** to learn continuous probability flows on the geometric manifold.

## Key Features
-   **SE(3) Equivariance**: Ensures physical validity of generated structures under rotation and translation.
-   **Optimal Transport**: Utilizes Minibatch Optimal Transport to construct geodesic flow paths, improving training stability and convergence.
-   **Continuous Normalizing Flows**: Solves the Neural ODE $dx/dt = v_\theta(x, t)$ for exact likelihood estimation and sampling.

## Project Structure
```
gfm/
├── src/gfm/
│   ├── core/           # Core algorithms (OT, Integration)
│   ├── layers/         # Equivariant Neural Network Layers
│   ├── models/         # Deep Learning Architectures
│   ├── data/           # Data Loading & Processing
│   └── experiments/    # Training Drivers
└── scripts/            # Execution Scripts
```


## Installation

It is recommended to use a virtual environment (conda or venv).

```bash
# Developer install
pip install -e .
```

## Usage

The package provides command-line interfaces for training, sampling, and analysis.

### 1. Training
Train the model on a single target molecule (overfitting/demonstration).
```bash
python scripts/train.py --smile "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --steps 5000 --device cuda
```

### 2. Sampling
Generate new conformers from a trained model.
```bash
python scripts/sample.py --model_path checkpoints/model_final.pt --n_samples 100 --output results/generated.xyz
```

### 3. Analysis
Evaluate stability (RMSD) and chemical validity.
```bash
python scripts/analyze.py --model_path checkpoints/model_final.pt --n_samples 50 --output_dir results/analysis
```

## Citation
If you use this code in your research, please cite:

```bibtex
@article{gfm2024,
  title={Geometric Flow Matching on Molecular Graphs},
  author={Research Team},
  journal={arXiv preprint arXiv:2402.XXXXX},
  year={2024}
}
```

