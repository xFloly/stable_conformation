import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
import torch
import yaml

from conformation_augmentation import perturb_structure
from conformation_similarity import calculate_rmsd

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_mol(path):
    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not load molecule: {path}")
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)
    return mol


def save_mol(mol, path):
    writer = Chem.SDWriter(path)
    for conf in mol.GetConformers():
        writer.write(mol, confId=conf.GetId())
    writer.close()



def generate_valid_augmented_set(input_dir, output_dir, amount, threshold, seed,
                                 enable_displacement, enable_rotation,
                                 perturbation_fraction,rotation_fraction,
                                   **perturb_kwargs):
    
    os.makedirs(output_dir, exist_ok=True)

    #SEED FOR REPRODUCTABILITY
    set_random_seed(seed)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mol'):
            ligand_id = os.path.splitext(filename)[0]
            ligand_path = os.path.join(input_dir, filename)
            
            try:
                original = load_mol(ligand_path)
            except Exception as e:
                print(f"{ligand_id}: Skipping due to error: {e}")
                continue

            conformers = []

            # Add the reference conformation
            conformers.append(original.GetConformer())

            accepted = 0
            attempts = 0
            max_attempts = amount * 100

            num_atoms = original.GetNumAtoms()
            atom_indices = np.arange(num_atoms)

            while accepted < amount and attempts < max_attempts:
                rmsd_val = 0.
                conformation = original

                if enable_displacement:
                    n_to_perturb = int(perturbation_fraction * num_atoms)
                    perturb_atoms_ids = np.random.choice(atom_indices, size=n_to_perturb, replace=False)
                else:
                    perturb_atoms_ids = None

                if enable_rotation:
                    n_to_rotate = int(rotation_fraction * num_atoms)
                    rotate_atoms_ids = np.random.choice(atom_indices, size=n_to_rotate, replace=False)
                else:
                    rotate_atoms_ids = None


                while threshold > rmsd_val:
                    perturbed = perturb_structure(conformation,enable_displacement=enable_displacement,enable_rotation=True, perturb_atoms_ids=perturb_atoms_ids, rotate_atoms_ids=rotate_atoms_ids, **perturb_kwargs)
                    rmsd_val = calculate_rmsd(original, perturbed)
                    conformation = perturbed 

     
                if threshold <= rmsd_val <= threshold+0.2:
                    conformers.append(perturbed.GetConformer())
                    print(f"{ligand_id}: Accepted conformer with RMSD {rmsd_val:.2f} (threshold: {threshold})")
                    accepted += 1
                    if accepted == amount:
                        break

                attempts += 1

            if accepted == 0:
                print(f"{ligand_id}: No conformers accepted (threshold: {threshold})")
                continue

            # Build multi-conformer molecule
            multi_mol = Chem.Mol(original)
            multi_mol.RemoveAllConformers()
            for conf in conformers:
                multi_mol.AddConformer(conf, assignId=True)

            output_path = os.path.join(output_dir, f"{ligand_id}.sdf")

            save_mol(multi_mol, output_path)
            print(f"{ligand_id}: {accepted} conformers accepted → saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate perturbed .mol structures using config.")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    


    generate_valid_augmented_set(**cfg)

if __name__ == "__main__":
    main()
