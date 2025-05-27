import os
import shutil
import yaml
import subprocess

TEMPLATE_CONFIG_PATH = "config.yaml"
MOL_INPUT_DIR = "../data/mol_subset"
OUTPUT_BASE_DIR = "../data/augmented"
TEMP_INPUT_DIR = "../temp_single_mol"

# Load base config
with open(TEMPLATE_CONFIG_PATH, 'r') as f:
    base_config = yaml.safe_load(f)

# Ensure temp input dir exists
os.makedirs(TEMP_INPUT_DIR, exist_ok=True)

# Iterate over .mol files
for filename in os.listdir(MOL_INPUT_DIR):
    if not filename.endswith(".mol"):
        continue

    ligand_id = os.path.splitext(filename)[0]
    mol_path = os.path.join(MOL_INPUT_DIR, filename)

    # Prepare temp input dir
    temp_mol_path = os.path.join(TEMP_INPUT_DIR, filename)
    shutil.copyfile(mol_path, temp_mol_path)

    # Set specific input/output paths in config
    new_config = base_config.copy()
    new_config["input_dir"] = TEMP_INPUT_DIR
    new_config["output_dir"] = os.path.join(OUTPUT_BASE_DIR, ligand_id)

    # Save updated config
    temp_config_path = f"./temp_config_{ligand_id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(new_config, f)

    # Run generate_fakes.py
    subprocess.run(["python", "generate_fakes.py", "--config", temp_config_path])

    # Clean up temp input/config file
    os.remove(temp_mol_path)
    os.remove(temp_config_path)
