from utils.dataset import MoleculeDataset

val_dset = MoleculeDataset('./data/val_Drugs.pkl')
data = val_dset[0]

print(data)           # Check if .pos exists
print(data.pos)       # Should be [num_atoms, 3]
