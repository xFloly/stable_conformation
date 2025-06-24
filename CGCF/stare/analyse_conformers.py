import torch
from torch_geometric.data import Batch
from models.edgecnf import EdgeCNF, simple_generate_batch
from models.cnf_edge import add_spectral_norm
from utils.misc import seed_all
from utils.dataset import MoleculeDataset
from utils.transforms import get_standard_transforms

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
checkpoint = torch.load('./models/ckpt_drugs.pt', map_location=device)
args = checkpoint['args']
seed_all(args.seed)

# Load validation dataset
tf = get_standard_transforms(order=args.aux_edge_order)
args.val_dataset = './data/val_Drugs.pkl'

val_dset = MoleculeDataset(args.val_dataset, transform=tf)

# Initialize model
model = EdgeCNF(args).to(device)
if args.spectral_norm:
    add_spectral_norm(model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Pick a molecule
data = val_dset[0].to(device)
batch = Batch.from_data_list([data])

# Compute edge lengths from original conformation
def compute_edge_lengths(data):
    src, dst = data.edge_index
    pos_src = data.pos[src]
    pos_dst = data.pos[dst]
    edge_lengths = torch.norm(pos_src - pos_dst, dim=1)
    return edge_lengths

edge_lengths_real = compute_edge_lengths(data).view(-1, 1)
logp_real = model.get_log_prob(batch, edge_lengths_real)
print(f"Original conformation log-likelihood: {logp_real.mean().item():.4f}")

# Sample fake conformers
num_samples = 10
samples, _ = model.sample(batch, num_samples=num_samples)  # shape: (E, num_samples)

# Score each sampled conformer
for i in range(num_samples):
    edge_lengths = samples[:, i].view(-1, 1)
    logp = model.get_log_prob(batch, edge_lengths)
    print(f"Sample {i+1:02d}: Log-likelihood = {logp.mean().item():.4f}")
