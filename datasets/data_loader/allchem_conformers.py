import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.rdmolfiles import SDWriter
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # wyłączenie warningów z RDKit


def log(msg, filename, start_time):
    elapsed = time.time() - start_time
    print(f"{filename} [{elapsed:.1f}s]: {msg}")


def generate_pruned_diverse_confs(
    mol,
    filename,
    num_output_confs=20,
    overgen_factor=3,
    base_prune=0.1,
    rmsd_bin_size=0.2,
    max_seconds=20,
    max_attempts=50
):
    mol = Chem.AddHs(mol)
    used_bins = set()
    selected_confs = []
    rejected_confs = []

    max_rmsd = rmsd_bin_size * (num_output_confs - 1)
    final_mol = Chem.AddHs(mol)
    final_mol.RemoveAllConformers()

    start_time = time.time()

    try:
        orig_conf = mol.GetConformer()
        final_mol.AddConformer(orig_conf, assignId=True)
        used_bins.add(0)
        selected_confs.append(0)
        log("→ Dodano oryginalny konformer (bin=0)", filename, start_time)
    except:
        log("[!] Nie udało się uzyskać oryginalnego konformera.", filename, start_time)
        return None, []

    attempt = 0
    prune_step = 0

    while (
        attempt < max_attempts and
        (time.time() - start_time) < max_seconds and
        len(selected_confs) < num_output_confs
    ):
        prune_step += 1
        if prune_step > num_output_confs:
            prune_step = 1
        prune_thresh = base_prune * prune_step
        attempt += 1

        temp_mol = Chem.Mol(mol)
        conf_ids = AllChem.EmbedMultipleConfs(
            temp_mol,
            numConfs=overgen_factor,
            pruneRmsThresh=prune_thresh,
            randomSeed=attempt,
        )

        for cid in conf_ids:
            try:
                conf = temp_mol.GetConformer(cid)
                coords = conf.GetPositions()

                # try:
                #     opt_mol = Chem.Mol(temp_mol)
                #     if AllChem.MMFFHasAllMoleculeParams(opt_mol):
                #         AllChem.MMFFOptimizeMolecule(opt_mol, confId=cid)
                #     else:
                #         AllChem.UFFOptimizeMolecule(opt_mol, confId=cid)
                #     coords = opt_mol.GetConformer(cid).GetPositions()
                # except:
                #     pass

                conf_new = Chem.Conformer(final_mol.GetNumAtoms())
                for i in range(final_mol.GetNumAtoms()):
                    x, y, z = coords[i]
                    conf_new.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
                new_cid = final_mol.AddConformer(conf_new, assignId=True)

                rmsd = rdMolAlign.GetBestRMS(final_mol, final_mol, prbId=new_cid, refId=0)
                bin_index = int(rmsd // rmsd_bin_size)

                if rmsd > max_rmsd:
                    continue

                if bin_index not in used_bins:
                    selected_confs.append(new_cid)
                    used_bins.add(bin_index)
                    # log(f"→ Konformer {new_cid}: RMSD={rmsd:.3f} Å, bin={bin_index} [✓✓✓✓] Dodany", filename, start_time)
                else:
                    rejected_confs.append(new_cid)

                if len(selected_confs) >= num_output_confs:
                    break

            except Exception as e:
                log(f"[!] Błąd przy przetwarzaniu konformera: {e}", filename, start_time)
                continue

    # Uzupełnianie brakujących konformerów losowo
    if len(selected_confs) < num_output_confs:
        missing = num_output_confs - len(selected_confs)
        log(f"[!] Brakuje {missing} konformerów – uzupełniam losowo z odrzuconych", filename, start_time)
        np.random.shuffle(rejected_confs)
        for new_cid in rejected_confs[:missing]:
            selected_confs.append(new_cid)
            log(f"→ Dodano losowy konformer {new_cid} (bin duplikat)", filename, start_time)

    log(f"[DEBUG] Wybrano {len(selected_confs)} konformerów: {selected_confs}", filename, start_time)
    log(f"[DEBUG] Odrzucone (do losowania): {len(rejected_confs)} konformerów", filename, start_time)

    return final_mol, selected_confs




def process_file(args):
    path, output_base, num_confs_target = args
    filename = os.path.basename(path)
    base = os.path.splitext(filename)[0]

    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        print(f"{base}: [!] Nie udało się wczytać pliku.")
        return

    print(f"\n== Przetwarzam: {filename} ==")
    mol_with_confs, conf_ids = generate_pruned_diverse_confs(
        mol,
        filename=base,
        num_output_confs=num_confs_target,
        overgen_factor=10,
        base_prune=0.1,
        rmsd_bin_size=0.1,
        max_seconds=300,
        max_attempts=100
    )

    if mol_with_confs is None or len(conf_ids) == 0:
        print(f"{base}: [!] Pominięto – brak konformerów.")
        return

    output_dir = os.path.join(output_base, base)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}.sdf")

    writer = SDWriter(output_path)
    for cid in conf_ids:
        try:
            rmsd = rdMolAlign.GetBestRMS(mol_with_confs, mol_with_confs, prbId=cid, refId=0)
            mol_with_confs.SetProp("RMSD_from_0", f"{rmsd:.3f}")
            writer.write(mol_with_confs, confId=cid)
            mol_with_confs.ClearProp("RMSD_from_0")
        except:
            continue
    writer.close()
    print(f"{base}: [✓] Zapisano {len(conf_ids)} konformerów do {output_path}")


# === Uruchomienie równoległe ===
def main():
    input_dir = "./data_loader/data/mol_subset"
    output_base = "augmented"
    num_confs_target = 20

    input_paths = [
        (os.path.join(input_dir, fname), output_base, num_confs_target)
        for fname in os.listdir(input_dir)
        if fname.endswith(".mol")
    ]

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_file, input_paths)


if __name__ == "__main__":
    main()
