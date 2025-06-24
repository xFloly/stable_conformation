import os
import shutil
import yaml
import subprocess
from multiprocessing import Pool, cpu_count

TEMPLATE_CONFIG_PATH = "./config.yaml"
MOL_INPUT_DIR = "./data/mol_subset"
OUTPUT_BASE_DIR = "./data/augmented_threshold_1_8-2_0"
TEMP_BASE_DIR = "./temp_jobs"

# Load base config once
with open(TEMPLATE_CONFIG_PATH, 'r') as f:
    base_config = yaml.safe_load(f)

os.makedirs(TEMP_BASE_DIR, exist_ok=True)


def process_mol(filename):
    if not filename.endswith(".mol"):
        return

    ligand_id = os.path.splitext(filename)[0]
    mol_path = os.path.join(MOL_INPUT_DIR, filename)

    # Unique temp dirs
    temp_input_dir = os.path.join(TEMP_BASE_DIR, f"{ligand_id}_input")
    temp_config_path = os.path.join(TEMP_BASE_DIR, f"config_{ligand_id}.yaml")

    os.makedirs(temp_input_dir, exist_ok=True)

    try:
        temp_mol_path = os.path.join(temp_input_dir, filename)
        shutil.copyfile(mol_path, temp_mol_path)

        new_config = base_config.copy()
        new_config["input_dir"] = temp_input_dir
        new_config["output_dir"] = os.path.join(OUTPUT_BASE_DIR, ligand_id)

        with open(temp_config_path, 'w') as f:
            yaml.dump(new_config, f)

        print(f"[{ligand_id}] Starting...")
        subprocess.run(["python", "generate_fakes.py", "--config", temp_config_path], check=True)
        print(f"[{ligand_id}] Done.")

    except Exception as e:
        print(f"[{ligand_id}] Error: {e}")

    finally:
        # Clean up
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    mol_files = os.listdir(MOL_INPUT_DIR)
    mol_files = [f for f in mol_files if f.endswith(".mol")]

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_mol, mol_files)
