import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

from conformation_augmentation import perturb_structure
from conformation_similarity import calculate_rmsd


def load_mol(path):
    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not load molecule: {path}")
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)
    return mol


def save_mol(mol, path):
    Chem.MolToMolFile(mol, path)


def generate_valid_augmented_set(input_dir, output_dir, amount, rmsd_threshold):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mol'):
            ligand_id = os.path.splitext(filename)[0]
            ligand_folder = os.path.join(output_dir, ligand_id)
            os.makedirs(ligand_folder, exist_ok=True)

            # Load original
            mol_path = os.path.join(input_dir, filename)
            original = load_mol(mol_path)

            # Save correct version
            correct_path = os.path.join(ligand_folder, 'correct.mol')
            save_mol(original, correct_path)

            accepted = 0
            attempts = 0
            max_attempts = amount * 5

            while accepted < amount and attempts < max_attempts:
                perturbed = perturb_structure(original)
                rmsd_val = calculate_rmsd(original, perturbed)

                if rmsd_val <= rmsd_threshold:
                    incorrect_path = os.path.join(ligand_folder, f'incorrect_{accepted+1:03d}.mol')
                    save_mol(perturbed, incorrect_path)
                    accepted += 1

                attempts += 1

            print(f"{ligand_id}: {accepted} accepted (threshold: {rmsd_threshold})")


def main():
    parser = argparse.ArgumentParser(description="Generate perturbed .mol structures with similarity constraint.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to .mol input files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output ligand folders.')
    parser.add_argument('--amount', type=int, default=3, help='Number of accepted perturbed conformations per molecule.')
    parser.add_argument('--threshold', type=float, default=1.5, help='Maximum allowed RMSD between original and perturbed.')

    args = parser.parse_args()

    generate_valid_augmented_set(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        amount=args.amount,
        rmsd_threshold=args.threshold
    )


if __name__ == '__main__':
    main()


#python generate_fakes.py   --input_dir ..\data\moles --output_dir ..\data\dataset
