import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, device):
        super(Dataset, self).__init__()
        self.df = dataframe
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user = torch.tensor(row[0], dtype=torch.long, device=self.device)
        item = torch.tensor(row[1], dtype=torch.long, device=self.device)
        label = torch.tensor(row[2], dtype=torch.float, device=self.device)

        return user, item, label
