import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from typing import List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

class GraphData:
    """
    Structured container for Graph Data.
    """
    def __init__(self, x: torch.Tensor, adj: torch.Tensor, atom_types: Optional[torch.Tensor] = None):
        self.x = x
        self.adj = adj
        self.atom_types = atom_types
        self.num_nodes = x.shape[0]
        self.dim = x.shape[1]

    def to(self, device):
        self.x = self.x.to(device)
        self.adj = self.adj.to(device)
        if self.atom_types is not None:
             self.atom_types = self.atom_types.to(device)
        return self

def smile_to_graph(smile: str) -> GraphData:
    """
    Converts a SMILES string to a GraphData object with generated 3D coordinates.
    Uses RDKit for conformer generation (ETKDG / MMFF).
    """
    if Chem is None:
        raise ImportError("RDKit is required for molecular processing.")
        
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    
    # 3D Conformer Generation
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res != 0:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
             raise ValueError(f"Failed to embed molecule: {smile}")
             
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass # Optimization optional if embedding succeeded
    
    # Extract Positions
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    x = torch.tensor(pos, dtype=torch.float32)
    
    # Center Coordinates to Origin (Standard Preprocessing)
    mean = x.mean(dim=0)
    x = x - mean
    
    # Scale to Unit Variance (Optimization for Neural Networks)
    # Stores scale factor to reconstruct original geometry later
    scale = x.std() + 1e-6
    x = x / scale
    
    # Extract Adjacency
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.tensor(adj, dtype=torch.float32) + torch.eye(adj.shape[0]) # Self-loops
    
    # Extract Features (Atomic Numbers)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_types = torch.tensor(atoms, dtype=torch.long)
    
    g = GraphData(x, adj, atom_types)
    g.norm_scale = scale # Save for unscaling
    return g

class MolecularDataset(Dataset):
    """
    PyTorch Dataset for Molecular Graphs.
    """
    def __init__(self, smiles_list: List[str]):
        self.graphs = []
        for s in smiles_list:
            try:
                self.graphs.append(smile_to_graph(s))
            except Exception as e:
                print(f"Warning: Skipped {s} due to error: {e}")
                
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
