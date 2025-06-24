import os
import shutil
import yaml
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

TEMPLATE_CONFIG_PATH = "config.yaml"
MOL_INPUT_DIR = "./data/mol_subset"
OUTPUT_BASE_DIR = "./data/augmented"
TEMP_INPUT_DIR = "./temp_single_mol"

def process_ligand(filename, base_config):
    if not filename.endswith(".mol"):
        return

    ligand_id = os.path.splitext(filename)[0]
    mol_path = os.path.join(MOL_INPUT_DIR, filename)

    # Create unique temp input dir per ligand
    ligand_temp_dir = os.path.join(TEMP_INPUT_DIR, ligand_id)
    os.makedirs(ligand_temp_dir, exist_ok=True)

    # Copy mol file to its own temp folder
    temp_mol_path = os.path.join(ligand_temp_dir, filename)
    shutil.copyfile(mol_path, temp_mol_path)

    # Set ligand-specific config
    new_config = base_config.copy()
    new_config["input_dir"] = ligand_temp_dir
    new_config["output_dir"] = os.path.join(OUTPUT_BASE_DIR, ligand_id)

    # Save temporary config
    temp_config_path = f"./temp_config_{ligand_id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(new_config, f)

    # Run generation
    result = subprocess.run(["python", "generate_fakes.py", "--config", temp_config_path])

    # Clean up
    shutil.rmtree(ligand_temp_dir, ignore_errors=True)
    os.remove(temp_config_path)

    return ligand_id if result.returncode == 0 else f"{ligand_id} (FAILED)"


if __name__ == "__main__":
    # Load base config
    with open(TEMPLATE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # Ensure necessary directories exist
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Copy original config to output dir once
    used_config_path = os.path.join(OUTPUT_BASE_DIR, "used_config.yaml")
    if not os.path.exists(used_config_path):
        shutil.copyfile(TEMPLATE_CONFIG_PATH, used_config_path)

    # Gather all .mol filenames
    mol_files = [f for f in os.listdir(MOL_INPUT_DIR) if f.endswith(".mol")]

    # Use multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_ligand, fname, base_config) for fname in mol_files]
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"✅ Done: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")
