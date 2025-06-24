from pathlib import Path
import os
import random
import traceback
import torch
import pandas as pd
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from models.cnf_edge.spectral_norm import add_spectral_norm
from scipy.stats import pearsonr, spearmanr

from models.edgecnf import EdgeCNF
from utils.misc import seed_all
from utils.transforms import get_standard_transforms
from data_loader.conformation_similarity import calculate_aligned_rmsd

# ----- CONFIG -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mol_dir = './data/mol_subset/'
multi_augmented_root = './data/augmented_rmsd_0_1-2_every_0_2/'
num_samples = None  # Set to None to process all, or an integer

# ----- LOAD MODEL -----
checkpoint = torch.load('./models/ckpt_drugs.pt', map_location=device)
args = checkpoint['args']
seed_all(args.seed)
tf = get_standard_transforms(order=args.aux_edge_order)

model = EdgeCNF(args).to(device)
if args.spectral_norm:
    add_spectral_norm(model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def mol_to_data_obj(mol):
    mol = copy.deepcopy(mol)

    atom_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()

    # Walidacja atomów
    if not atom_feats:
        raise ValueError("Molecule has no atoms.")

    if min(atom_feats) <= 0:
        raise ValueError(f"Invalid atom types found: {atom_feats}")

    if hasattr(args, "num_atom_types") and max(atom_feats) >= args.num_atom_types:
        raise ValueError(f"Atom type {max(atom_feats)} exceeds allowed maximum {args.num_atom_types}.")

    edge_index = []
    edge_feats = []

    bond_type_to_int = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_type_to_int.get(bond.GetBondType(), -1)

        if bond_feat == -1:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")

        edge_index.append((i, j))
        edge_feats.append(bond_feat)
        edge_index.append((j, i))
        edge_feats.append(bond_feat)

    if not edge_index:
        raise ValueError("No edges found in molecule.")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.long)

    # Walidacja indeksów
    if edge_index.max().item() >= len(atom_feats):
        raise ValueError("Edge index out of bounds.")

    data = Data(
        x=torch.tensor(atom_feats, dtype=torch.long).view(-1, 1),
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=edge_index,
        edge_type=edge_feats
    )
    data.node_type = data.x.squeeze(-1)

    if tf:
        data = tf(data)
    return data


def compute_edge_lengths(data):
    src, dst = data.edge_index
    pos_src = data.pos[src]
    pos_dst = data.pos[dst]
    return torch.norm(pos_src - pos_dst, dim=1)

# Iterate through all augmentation folders
augmented_folders = sorted([f for f in Path(multi_augmented_root).iterdir() if f.is_dir()])

for augmented_dir_path in augmented_folders:
    augmented_dir = str(augmented_dir_path)
    print(f"\n=== Processing {augmented_dir} ===\n")

    # Metrics for current folder
    total = 0
    top1_count = 0
    top3_count = 0
    top5_count = 0
    sum_ranks = 0
    z_scores = []
    results = []

    mol_files = [f for f in os.listdir(mol_dir) if f.endswith('.mol')]
    if num_samples:
        random.shuffle(mol_files)
        mol_files = mol_files[:num_samples]

    for filename in mol_files:
        ligand_id = os.path.splitext(filename)[0]
        original_path = os.path.join(mol_dir, filename)
        augmented_path = os.path.join(augmented_dir, ligand_id, f"{ligand_id}.sdf")

        if not os.path.exists(augmented_path):
            print(f"Missing: {augmented_path}")
            continue

        supplier = Chem.SDMolSupplier(augmented_path, removeHs=False)
        mols = [mol for mol in supplier if mol is not None]

        if not mols:
            print(f"Could not parse {augmented_path} or no valid molecules found.")
            continue

        print(f"\n{ligand_id} — {len(mols)} conformations")
        logps = []
        rmsds = []
        energies = []

        for conf_id, mol in enumerate(mols):
            try:
                data = mol_to_data_obj(mol).to(device)
                batch = Batch.from_data_list([data])
                edge_lengths = compute_edge_lengths(data).view(-1, 1)
                logp = model.get_log_prob(batch, edge_lengths).mean().item()
            except Exception as e:
                print(f" → Skipped {ligand_id} conf {conf_id} due to logp error: {e}")
                traceback.print_exc()
                continue

            try:
                rmsd = calculate_aligned_rmsd(mols[0], mol)
            except Exception as e:
                print(f" → RMSD error ({ligand_id} conf {conf_id}): {e}")
                rmsd = float('nan')

            try:
                mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                if ff is None:
                    raise ValueError("Force field could not be initialized")
                energy = ff.CalcEnergy()
            except Exception as e:
                print(f" → MMFF Energy error ({ligand_id} conf {conf_id}): {e}")
                energy = float('nan')


            print(f"{conf_id:>10}: Log-likelihood = {logp:.4f}")
            print(f" → RMSD = {rmsd:.4f}")
            print(f" → MMFF Energy = {energy:.4f}")

            results.append({
                'ligand_id': ligand_id,
                'conf_id': conf_id,
                'logp': logp,
                'rmsd': rmsd,
                'energy': energy
            })

            logps.append(logp)
            rmsds.append(rmsd)
            energies.append(energy)

        # Skip ranking and correlation calculations if no logps
        if not logps:
            continue

        total += 1
        original_logp = logps[0]
        sorted_logps = sorted(logps, reverse=True)
        rank = sorted_logps.index(original_logp) + 1
        top1 = rank == 1
        top3 = rank <= 3
        top5 = rank <= 5
        mean_logp = sum(logps) / len(logps)
        std_logp = (sum((x - mean_logp) ** 2 for x in logps) / len(logps)) ** 0.5
        z_score = (original_logp - mean_logp) / std_logp if std_logp > 0 else 0.0
        next_best = max(logps[1:])
        gap = original_logp - next_best

        sum_ranks += rank
        top1_count += top1
        top3_count += top3
        top5_count += top5
        z_scores.append(z_score)

    output_csv = f"results_{augmented_dir_path.name}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nZapisano dane do '{output_csv}'")
