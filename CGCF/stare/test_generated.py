from rdkit import Chem
from rdkit.Chem import Draw

# Load generated molecules
import pickle
with open("generated.pkl", "rb") as f:
    molecules = pickle.load(f)

# Draw first 5 molecules
Draw.MolsToGridImage(molecules[:5], molsPerRow=5, subImgSize=(200,200)).show()

for i, mol in enumerate(molecules[:5]):
    print(f"Molecule {i + 1}:")
    print(f"  SMILES: {Chem.MolToSmiles(mol)}")
    print(f"  Num atoms: {mol.GetNumAtoms()}")

from rdkit import Chem

for i, mol in enumerate(molecules[:3]):
    conf = mol.GetConformer()
    print(f"Conformer {i + 1} has {conf.GetNumAtoms()} atoms.")
    print("First 3D coordinates:")
    for atom_idx in range(min(3, conf.GetNumAtoms())):
        pos = conf.GetAtomPosition(atom_idx)
        print(f"  Atom {atom_idx}: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
    print("---")
