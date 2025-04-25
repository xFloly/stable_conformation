import os
import argparse
import pymol2  # pymol2 is the PyMOL python API

def convert_cif_to_mol(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    with pymol2.PyMOL() as pymol:
        pymol.start()
        
        for filename in os.listdir(input_path):
            if filename.endswith('.cif'):
                input_file = os.path.join(input_path, filename)
                output_filename = os.path.splitext(filename)[0] + '.mol'
                output_file = os.path.join(output_path, output_filename)

                try:
                    obj_name = os.path.splitext(filename)[0]
                    pymol.cmd.load(input_file, obj_name)
                    pymol.cmd.save(output_file, obj_name, format='mol')
                    pymol.cmd.delete(obj_name)

                    print(f"Converted {filename} to {output_filename}")

                except Exception as e:
                    print(f"Error processing {input_file}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CIF files to MOL format.")
    parser.add_argument("--input_path", type=str, help="Path to the input folder containing CIF files.")
    parser.add_argument("--output_path", type=str, help="Path to the output folder for MOL files.")

    args = parser.parse_args()

    convert_cif_to_mol(args.input_path, args.output_path)

#python cif_to_mol.py --input_path ../data/cifs --output_path ../data/mols