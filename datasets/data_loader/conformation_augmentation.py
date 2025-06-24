import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import TransformConformer
from scipy.spatial.transform import Rotation as R

def get_coordinates(mol: Chem.Mol) -> np.ndarray:
    """Extract 3D coordinates from a molecule."""
    conformer = mol.GetConformer()
    coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

def set_coordinates(mol: Chem.Mol, new_coords: np.ndarray):
    """Overwrite coordinates in a molecule's conformer."""
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, new_coords[i])

def perturb_structure(
    mol: Chem.Mol,
    enable_displacement: bool = True,
    enable_rotation: bool = True,
    max_displacement: float = 0.05,
    max_displacement_h: float = 0.02,
    max_rotation_angle: float = 5.0,     # degrees
    # perturbation_fraction: float = 0.15, # fraction of atoms displaced
    # rotation_fraction: float = 0.25,      # fraction of atoms rotated
    enable_random_choice: bool = True,
    perturb_atoms_ids: list[int] = None,
    rotate_atoms_ids: list[int] = None 
) -> Chem.Mol:
    """
    Apply random perturbations to a RDKit molecule.

    Args:
        mol: RDKit Mol object with 3D conformer
        enable_displacement: Whether to apply random displacements
        enable_rotation: Whether to apply random rotation
        max_displacement: Max displacement (Å) for heavy atoms
        max_displacement_h: Max displacement (Å) for hydrogen atoms
        max_rotation_angle: Max rotation angle (degrees)
        perturbation_fraction: Fraction of atoms to displace
        rotation_fraction: Fraction of atoms to rotate

    Returns:
        new_mol: RDKit Mol with modified conformer
    """
    new_mol = Chem.Mol(mol)
    coords = get_coordinates(new_mol)
    num_atoms = new_mol.GetNumAtoms()
    # atom_indices = np.arange(num_atoms)

    if enable_displacement:
        selected_indices = perturb_atoms_ids

        for atom_idx in selected_indices:
            atom = new_mol.GetAtomWithIdx(int(atom_idx))
            disp = np.random.uniform(
                -max_displacement_h if atom.GetSymbol() == 'H' else -max_displacement,
                 max_displacement_h if atom.GetSymbol() == 'H' else  max_displacement,
                size=3
            )
            coords[atom_idx] += disp

    if enable_rotation:
        selected_indices = rotate_atoms_ids

        angle_deg = np.random.uniform(-max_rotation_angle, max_rotation_angle)
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = R.from_rotvec(np.deg2rad(angle_deg) * axis)

        coords[selected_indices] = rotation.apply(coords[selected_indices])

    set_coordinates(new_mol, coords)
    return new_mol



