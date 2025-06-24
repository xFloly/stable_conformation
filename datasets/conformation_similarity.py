import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

def get_coordinates(mol: Chem.Mol) -> np.ndarray:
    """
    Extracts 3D coordinates from the first conformer of a molecule.
    """
    conformer = mol.GetConformer()
    coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

def calculate_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Computes RMSD directly between two RDKit molecules' atomic positions.
    
    Assumes same atom ordering.
    """
    coords1 = get_coordinates(mol1)
    coords2 = get_coordinates(mol2)
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def calculate_aligned_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Aligns mol2 to mol1 using Kabsch algorithm, then computes RMSD.
    """
    rmsd_val = rdMolAlign.GetBestRMS(mol1, mol2)
    return rmsd_val

def calculate_centroid_distance(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Returns the Euclidean distance between the centroids (center of mass) of two molecules.
    """
    coords1 = get_coordinates(mol1)
    coords2 = get_coordinates(mol2)

    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)

    return np.linalg.norm(centroid1 - centroid2)

def calculate_displacement_stats(mol1: Chem.Mol, mol2: Chem.Mol) -> dict:
    """
    Computes per-atom displacement statistics between two molecules.
    """
    coords1 = get_coordinates(mol1)
    coords2 = get_coordinates(mol2)

    displacements = np.linalg.norm(coords1 - coords2, axis=1)
    return {
        'mean_displacement': float(np.mean(displacements)),
        'max_displacement': float(np.max(displacements)),
        'min_displacement': float(np.min(displacements)),
    }
