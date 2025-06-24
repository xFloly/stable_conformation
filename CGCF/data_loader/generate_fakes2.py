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


def generate_valid_augmented_set(input_dir, output_dir, amount, max_rmsd, seed, **perturb_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    set_random_seed(seed)

    bin_edges = np.linspace(0.1, max_rmsd, amount + 1)  # np. 20 zakresów RMSD

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
            rmsd_values = []
            conformers.append(original.GetConformer())
            rmsd_values.append(0.0)  # RMSD względem siebie = 0.0

            found_bins = [False] * amount
            attempts = 0
            max_attempts = 2000

            while not all(found_bins) and attempts < max_attempts:
                perturbed = perturb_structure(original, **perturb_kwargs)
                rmsd_val = calculate_rmsd(original, perturbed)

                for i in range(amount):
                    if not found_bins[i] and bin_edges[i] <= rmsd_val < bin_edges[i + 1]:
                        conformers.append(perturbed.GetConformer())
                        rmsd_values.append(rmsd_val)
                        found_bins[i] = True
                        print(f"{ligand_id}: Found RMSD {rmsd_val:.2f} for bin {i+1}/{amount}")
                        break

                attempts += 1

            if sum(found_bins) < amount:
                print(f"{ligand_id}: Only {sum(found_bins)} bins filled (target: {amount})")

            # Zapisz do SDF
            multi_mol = Chem.Mol(original)
            multi_mol.RemoveAllConformers()
            for conf in conformers:
                multi_mol.AddConformer(conf, assignId=True)

            output_path = os.path.join(output_dir, f"{ligand_id}.sdf")
            save_mol(multi_mol, output_path)

            # Wypisz RMSDy
            print(f"\n{ligand_id}: Saved {len(conformers)} conformers to {output_path}")
            print("RMSD values:")
            for i, rmsd in enumerate(rmsd_values):
                print(f"  Conf {i:2d}: RMSD = {rmsd:.3f}")
            print("-" * 40)



def main():
    parser = argparse.ArgumentParser(description="Generate perturbed .mol structures using config.")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    


    generate_valid_augmented_set(**cfg)

if __name__ == "__main__":
    main()
