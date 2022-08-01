import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.to_numpy()
        self.target = target.to_numpy()
        self.columns = data.columns

    def __getitem__(self, index):
        random_idx = np.random.randint(0, len(self))
        x_c = torch.tensor(self.data[random_idx], dtype=torch.float)
        x = torch.tensor(self.data[index], dtype=torch.float)

        return x, x_c

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    def min(self):
        return torch.tensor(self.data.min(axis=0), dtype=torch.float)

    def max(self):
        return torch.tensor(self.data.max(axis=0), dtype=torch.float)
