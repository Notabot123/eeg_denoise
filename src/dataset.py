# used in training job
import os
import numpy as np
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        noisy = torch.tensor(data["noisy"], dtype=torch.float32)
        clean = torch.tensor(data["clean"], dtype=torch.float32)
        return noisy, clean
