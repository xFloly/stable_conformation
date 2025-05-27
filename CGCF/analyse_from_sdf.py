import traceback
import os
import random
import torch
import copy
from rdkit import Chem
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
augmented_dir = './data/augmented/'
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

        edge_index.append((i, j))
        edge_feats.append(bond_feat)
        edge_index.append((j, i))
        edge_feats.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.long)

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

# ----- METRICS ----- #
total = 0
top1_count = 0
top3_count = 0
top5_count = 0
sum_ranks = 0
z_scores = []

# ----- Process All Molecules -----
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

    # Load all conformations (original + augmented)
    supplier = Chem.SDMolSupplier(augmented_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    if not mols:
        print(f"Could not parse {augmented_path} or no valid molecules found.")
        continue

    print(f"\n{ligand_id} — {len(mols)} conformations")
    logps = []
    rmsds = []
    for conf_id, mol in enumerate(mols):
        try:
            data = mol_to_data_obj(mol).to(device)
            batch = Batch.from_data_list([data])
            edge_lengths = compute_edge_lengths(data).view(-1, 1)
            logp = model.get_log_prob(batch, edge_lengths)
            logps.append(logp.mean().item())
            tag = "Original" if conf_id == 0 else f"Conf {conf_id}"
            print(f"{tag:>10}: Log-likelihood = {logp.mean().item():.4f}")
            if conf_id != 0:
                original = mols[0]
                rmsd = calculate_aligned_rmsd(original, mol)
                rmsds.append(rmsd)
                print(f" → Comparing to Original: {rmsd:.4f}")

        except Exception as e:
            print(f"Error in {ligand_id} conf {conf_id}: {e}")
            traceback.print_exc()
            logps.append(float('-inf'))

    # ----- METRICS ----- #
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

    print(f" → Rank of Original: {rank}")
    print(f" → Z-score: {z_score:.3f}")
    print(f" → Log-likelihood Gap to Next Best: {gap:.3f}")

    # Accumulate stats
    total += 1
    sum_ranks += rank
    top1_count += top1
    top3_count += top3
    top5_count += top5
    z_scores.append(z_score)

    spearman, _ = spearmanr(logps[1:], rmsds)
    pearson, _ = pearsonr(logps[1:], rmsds)
    print(f" → Spearman Correlation: {spearman:.3f}, Pearson Correlation: {pearson:.3f}")

if total > 0:
    avg_rank = sum_ranks / total
    avg_z = sum(z_scores) / total
    print("\n===== SUMMARY =====")
    print(f"Total Molecules: {total}")
    print(f"Top-1 Accuracy: {top1_count / total:.2%}")
    print(f"Top-3 Accuracy: {top3_count / total:.2%}")
    print(f"Top-5 Accuracy: {top5_count / total:.2%}")
    print(f"Average Rank of Original: {avg_rank:.2f}")
    print(f"Average Z-score: {avg_z:.3f}")
