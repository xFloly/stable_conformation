import os
import argparse
import subprocess
from rdkit import Chem

def convert_cif_to_mol(cif_path, mol_path):
    """
    Converts a .cif file to .mol using Open Babel via command-line.
    """
    cmd = f'obabel -icif "{cif_path}" -omol -O "{mol_path}"'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        raise RuntimeError(f"[OpenBabel ERROR] Failed to convert {cif_path}:\n{result.stderr.decode().strip()}")

def validate_mol(mol_path):
    """
    Tries to load a .mol file using RDKit to validate it.
    """
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    if mol is None:
        raise ValueError(f"[RDKit ERROR] Cannot read {mol_path}")
    return mol

def batch_convert(input_dir, output_dir):
    """
    Converts all .cif files in a folder to .mol, verifies them with RDKit.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.cif')]

    for filename in files:
        cif_path = os.path.join(input_dir, filename)
        mol_name = os.path.splitext(filename)[0] + ".mol"
        mol_path = os.path.join(output_dir, mol_name)

        try:
            convert_cif_to_mol(cif_path, mol_path)
            validate_mol(mol_path)
            print(f"[OK] {filename} → {mol_name}")
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")
            # Optional: remove invalid mol file
            if os.path.exists(mol_path):
                os.remove(mol_path)

def main():
    parser = argparse.ArgumentParser(description="Batch convert .cif files to .mol using Open Babel + RDKit.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to folder with .cif files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save converted .mol files.")

    args = parser.parse_args()
    batch_convert(args.input_path, args.output_path)

if __name__ == "__main__":
    main()


#python scr/cif_to_mol.py --input_path ../data/cifs --output_path ../data/mols