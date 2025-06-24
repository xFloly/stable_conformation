import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# Example SMILES strings – you can replace/add more
smiles_list = [
    "CCO",     # ethanol
    "CC(=O)O", # acetic acid
    "c1ccccc1" # benzene
]

all_conformers = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)  # add hydrogens
    num_confs = 5  # generate 5 conformations per molecule
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)

    for conf_id in ids:
        single_conf = Chem.Mol(mol)  # clone the molecule
        conf = mol.GetConformer(conf_id)
        single_conf.RemoveAllConformers()
        single_conf.AddConformer(conf, assignId=True)
        all_conformers.append(single_conf)

# Save to pickle file
os.makedirs("data/qm9", exist_ok=True)
with open("data/qm9/test.pkl", "wb") as f:
    pickle.dump(all_conformers, f)

print(f"Saved {len(all_conformers)} conformers to data/qm9/test.pkl")
