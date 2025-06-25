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
from torch_scatter import scatter_mean
from scipy.stats import pearsonr, spearmanr

from models.cnf_edge.spectral_norm import add_spectral_norm
from models.edgecnf import EdgeCNF
from utils.misc import seed_all
from utils.transforms import get_standard_transforms
# from data_loader.conformation_similarity import calculate_aligned_rmsd
from conformation_similarity import calculate_aligned_rmsd

# ----- CONFIG -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mol_dir = './data/mol_subset/'
# multi_augmented_root = './data/augmented_rmsd_0_1-2_every_0_2/'
# multi_augmented_root = './data/aug_alchem'
multi_augmented_root = './data/new'
num_samples = None  # Set to None to process all, or an integer


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
    """Compute the distance between each pair of connected vertices in data."""
    src, dst = data.edge_index
    pos_src = data.pos[src]
    pos_dst = data.pos[dst]
    return torch.norm(pos_src - pos_dst, dim=1)

def compute_logps_for_all_conformers(mols, model, device):
    """
    Given a list of RDKit mol objects (mols), build a batched PyG Data object,
    run model.get_log_prob on the batch, and return a list of logp values
    corresponding to each conformer.

    IMPORTANT: If model.get_log_prob returns log-probabilities at the edge level,
    we must group them by graph with the correct index. Below, we assume that
    log_prob_output is shape [num_edges, 1] (or [num_edges]) and we gather indices
    via batch.edge_index[0], then do scatter_mean(...).
    If your model returns per-node or per-graph logs, you need to adjust accordingly.
    """
    data_list = []
    valid_idx_map = []

    for conf_id, mol in enumerate(mols):
        try:
            data = mol_to_data_obj(mol)
            data_list.append(data)
            valid_idx_map.append(conf_id)
        except Exception as e:
            print(f" → Skipped conf {conf_id} due to data error: {e}")

    if not data_list:
        return []

    # Combine into a single large batch
    batch = Batch.from_data_list(data_list).to(device)

    # Compute edge lengths for the entire batch
    edge_lengths = compute_edge_lengths(batch).view(-1, 1)

    # Get log probabilities from the model
    log_prob_output = model.get_log_prob(batch, edge_lengths)

    # If log_prob_output is shape [E, 1], remove trailing dim
    if len(log_prob_output.shape) == 2 and log_prob_output.shape[1] == 1:
        log_prob_output = log_prob_output.squeeze(1)

    # We now have one log-prob per edge. We need an array telling us
    # which graph each edge belongs to. The first row of batch.edge_index
    # is the "source" node. We look up its graph via batch.batch[source].
    edge2graph = batch.batch[batch.edge_index[0]]

    # Aggregate log-probs per graph
    logps_per_graph = scatter_mean(log_prob_output, edge2graph, dim=0)

    # Convert to Python floats
    logps = logps_per_graph.detach().cpu().tolist()

    # Re-insert them according to the original conf indices
    final_logps = [float('nan')] * len(mols)
    for offset, conf_id in enumerate(valid_idx_map):
        final_logps[conf_id] = logps[offset]

    return final_logps

# ----- MAIN PROCESSING LOOP -----

# ----- MAIN PROCESSING LOOP -----

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

    ligand_folders = sorted([f for f in Path(augmented_dir).iterdir() if f.is_dir()])
    missing_files = 0

    for ligand_folder in ligand_folders:
        ligand_id = ligand_folder.name
        augmented_path = os.path.join(augmented_dir, ligand_id, f"{ligand_id}.sdf")

        if not os.path.exists(augmented_path):
            print(f"[WARNING] Missing augmented SDF for ligand {ligand_id} at path: {augmented_path}")
            missing_files += 1
            continue

        supplier = Chem.SDMolSupplier(augmented_path, removeHs=False)
        mols = [mol for mol in supplier if mol is not None]

        if not mols:
            print(f"[WARNING] Could not parse {augmented_path} or no valid molecules found.")
            continue

        print(f"\n{ligand_id} — {len(mols)} conformations")

        # --- Batched logp computation for all conformers ---
        logps = compute_logps_for_all_conformers(mols, model, device)

        if not any(not pd.isna(lp) for lp in logps):
            continue

        for conf_id, mol in enumerate(mols):
            logp = logps[conf_id]
            if pd.isna(logp):
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

        # Skip ranking if no valid logps
        valid_logps = [lp for lp in logps if not pd.isna(lp)]
        if not valid_logps:
            continue

        total += 1

        original_logp = valid_logps[0]
        sorted_logps = sorted(valid_logps, reverse=True)
        rank = sorted_logps.index(original_logp) + 1
        top1 = rank == 1
        top3 = rank <= 3
        top5 = rank <= 5
        mean_logp = sum(valid_logps) / len(valid_logps)
        std_logp = (sum((x - mean_logp) ** 2 for x in valid_logps) / len(valid_logps)) ** 0.5
        z_score = (original_logp - mean_logp) / std_logp if std_logp > 0 else 0.0
        next_best = max(valid_logps[1:]) if len(valid_logps) > 1 else float('-inf')

        sum_ranks += rank
        top1_count += top1
        top3_count += top3
        top5_count += top5
        z_scores.append(z_score)

    print(f"\n[INFO] Total missing ligand SDFs in {augmented_dir_path.name}: {missing_files}/{len(ligand_folders)}")

    # Save results for this folder
    output_csv = f"results_{augmented_dir_path.name}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nZapisano dane do '{output_csv}'")
