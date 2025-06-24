import os
import random
import torch
import copy
from rdkit import Chem
from torch_geometric.data import Data, Batch

from models.edgecnf import EdgeCNF
from models.cnf_edge import add_spectral_norm
from utils.misc import seed_all
from utils.transforms import get_standard_transforms

# ----- CONFIG -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sdf_dir = './data/augmented/'  # folder z plikami .sdf
num_sdf_files = 10             # liczba analizowanych losowych plików .sdf

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

# ----- RDKit → PyG -----
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

# ----- ANALYZE SDF FILES -----
sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]
random.shuffle(sdf_files)
sdf_files = sdf_files[:num_sdf_files]

for sdf_name in sdf_files:
    sdf_path = os.path.join(sdf_dir, sdf_name)
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    conformers = [mol for mol in supplier if mol is not None]

    if not conformers:
        print(f"Could not parse {sdf_name} or no valid conformers found.")
        continue

    print(f"\n{sdf_name} — {len(conformers)} conformations")
    for conf_id, mol in enumerate(conformers):
        try:
            data = mol_to_data_obj(mol).to(device)
            batch = Batch.from_data_list([data])
            edge_lengths = compute_edge_lengths(data).view(-1, 1)
            logp = model.get_log_prob(batch, edge_lengths)
            tag = "Original" if conf_id == 0 else f"Conf {conf_id}"
            print(f"{tag:>10}: Log-likelihood = {logp.mean().item():.4f}")
        except Exception as e:
            print(f"Error in {sdf_name} conf {conf_id}: {e}")
