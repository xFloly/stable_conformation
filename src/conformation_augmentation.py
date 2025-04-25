import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import TransformConformer
from scipy.spatial.transform import Rotation as R

# ------------------------
# === AUGMENTATION FLAGS ===
# ------------------------

ENABLE_DISPLACEMENT = True
ENABLE_ROTATION = True

# ------------------------
# === AUGMENTATION PARAMS ===
# ------------------------

MAX_DISPLACEMENT = 0.05       # for heavy atoms
MAX_DISPLACEMENT_H = 0.02     # for hydrogen
MAX_ROTATION_ANGLE = 5.0      # degrees

# ------------------------
# === AUGMENTATION LOGIC ===
# ------------------------

def get_coordinates(mol: Chem.Mol) -> np.ndarray:
    """
    Extract 3D coordinates from a molecule.
    """
    conformer = mol.GetConformer()
    coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

def set_coordinates(mol: Chem.Mol, new_coords: np.ndarray):
    """
    Overwrites coordinates in a molecule's conformer.
    """
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, new_coords[i])

def perturb_structure(mol: Chem.Mol) -> Chem.Mol:
    """
    Apply random perturbations to a RDKit molecule.
    
    Args:
        mol: RDKit Mol object with 3D conformer
    
    Returns:
        new_mol: RDKit Mol with modified conformer
    """
    new_mol = Chem.Mol(mol)
    coords = get_coordinates(new_mol)

    if ENABLE_DISPLACEMENT:
        displacements = []
        for atom_idx in range(new_mol.GetNumAtoms()):
            atom = new_mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == 'H':
                disp = np.random.uniform(-MAX_DISPLACEMENT_H, MAX_DISPLACEMENT_H, size=3)
            else:
                disp = np.random.uniform(-MAX_DISPLACEMENT, MAX_DISPLACEMENT, size=3)
            displacements.append(disp)
        
        displacements = np.array(displacements)
        coords += displacements

    if ENABLE_ROTATION:
        angle_deg = np.random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)  # normalize rotation axis

        rotation = R.from_rotvec(np.deg2rad(angle_deg) * axis)
        coords = rotation.apply(coords)

    set_coordinates(new_mol, coords)
    return new_mol
