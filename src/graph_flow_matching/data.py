import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

class GraphData:
    """
    Container for graph data.
    x: Node features [N, D]
    adj: Adjacency matrix [N, N]
    """
    def __init__(self, x: torch.Tensor, adj: torch.Tensor):
        self.x = x
        self.adj = adj
        self.num_nodes = x.shape[0]
        self.dim = x.shape[1]

    def to(self, device):
        self.x = self.x.to(device)
        self.adj = self.adj.to(device)
        return self

def generate_grid_graph(grid_size: int = 10):
    """
    Generates a 2D grid graph.
    Returns: GraphData
    """
    G = nx.grid_2d_graph(grid_size, grid_size)
    adj = nx.adjacency_matrix(G).todense()
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Node features are their (normalized) positions in the grid
    nodes = list(G.nodes())
    x = torch.tensor(nodes, dtype=torch.float32)
    # Normalize to [-1, 1]
    x = (x / (grid_size - 1)) * 2 - 1
    
    return GraphData(x, adj)

def generate_ring_graph(num_nodes: int = 100, noise: float = 0.05):
    """
    Generates a ring graph with noisy positions.
    Returns: GraphData
    """
    G = nx.cycle_graph(num_nodes)
    adj = nx.adjacency_matrix(G).todense()
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Node features: positions on a circle
    theta = torch.linspace(0, 2*np.pi, num_nodes + 1)[:-1]
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    
    # Add noise
    x += torch.randn_like(x) * noise
    
    return GraphData(x, adj)

def smile_to_graph(smile: str):
    """
    Converts a SMILES string to a GraphData object with 3D coordinates.
    """
    if Chem is None:
        raise ImportError("RDKit is not installed.")
        
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    res = AllChem.EmbedMolecule(mol)
    if res != 0:
        # Fallback if embedding fails (try random coordinates or more attempts)
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
             raise ValueError(f"Could not generate conformer for SMILES: {smile}")
             
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass # Optimization might fail but we have coords
    
    # Get positions
    conf = mol.GetConformer()
    pos = conf.GetPositions() # [N, 3]
    x = torch.tensor(pos, dtype=torch.float32)
    
    # Center and scale simple normalization
    x = x - x.mean(dim=0)
    # Optional: scale? Let's keep angstrom scale or normalize to unit sphere?
    # For stability, let's normalize to roughly unit variance or fit in [-2, 2] box
    # scale = 1.0 
    # x = x / scale
    
    # Adjacency
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.tensor(adj, dtype=torch.float32) + torch.eye(adj.shape[0]) # Add self-loops
    
    # Atom features (Atomic Numbers)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_types = torch.tensor(atom_types, dtype=torch.long)
    
    g_data = GraphData(x, adj)
    g_data.atom_types = atom_types
    return g_data

class MolecularGraphDataset(Dataset):
    def __init__(self, smiles_list: list):
        self.graphs = []
        for smile in smiles_list:
            try:
                g = smile_to_graph(smile)
                self.graphs.append(g)
            except Exception as e:
                print(f"Skipping {smile}: {e}")
                
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

class GraphDataset(Dataset):
    """
    Simple dataset wrapper.
    For now, we just return the same graph with different noise instances if needed,
    or a list of graphs.
    """
    def __init__(self, graphs: list):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
