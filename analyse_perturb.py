import torch
import copy
from torch_geometric.data import Batch
from CGCF.models.edgecnf import EdgeCNF
from models.cnf_edge import add_spectral_norm
from utils.misc import seed_all
from utils.dataset import MoleculeDataset
from utils.transforms import get_standard_transforms

# ---------- Setup ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Load Model ----------
checkpoint = torch.load('./pretrained/ckpt_drugs.pt', map_location=device)
args = checkpoint['args']
seed_all(args.seed)

tf = get_standard_transforms(order=args.aux_edge_order)
args.val_dataset = './data/val_Drugs.pkl'

val_dset = MoleculeDataset(args.val_dataset, transform=tf)

model = EdgeCNF(args).to(device)
if args.spectral_norm:
    add_spectral_norm(model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# ---------- Edge Length Calculator ----------
def compute_edge_lengths(data):
    src, dst = data.edge_index
    pos_src = data.pos[src]
    pos_dst = data.pos[dst]
    edge_lengths = torch.norm(pos_src - pos_dst, dim=1)
    return edge_lengths

# ---------- Perturb Function ----------
def perturb_coordinates(data, noise_std=0.1):
    noisy_data = copy.deepcopy(data)
    noise = torch.randn_like(noisy_data.pos) * noise_std
    noisy_data.pos = noisy_data.pos + noise
    return noisy_data

# ---------- Pick Molecule ----------
data = val_dset[1].to(device)
batch = Batch.from_data_list([data])

# ---------- Score Original ----------
edge_lengths_real = compute_edge_lengths(data).view(-1, 1)
logp_real = model.get_log_prob(batch, edge_lengths_real)
print(f"Original conformation log-likelihood: {logp_real.mean().item():.4f}")

# ---------- Score Perturbations with Decreasing Noise ----------
print("\nPerturbed conformers with decreasing noise:")

noise_levels = [0.3, 0.2, 0.1, 0.05, 0.01, 0.001]

for noise_std in noise_levels:
    perturbed_data = perturb_coordinates(data, noise_std=noise_std).to(device)
    perturbed_batch = Batch.from_data_list([perturbed_data])
    edge_lengths_perturbed = compute_edge_lengths(perturbed_data).view(-1, 1)
    logp = model.get_log_prob(perturbed_batch, edge_lengths_perturbed)
    print(f"Noise STD {noise_std:>6}: Log-likelihood = {logp.mean().item():.4f}")
