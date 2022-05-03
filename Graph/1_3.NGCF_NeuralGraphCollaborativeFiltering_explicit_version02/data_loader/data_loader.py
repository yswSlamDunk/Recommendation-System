import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        super(Dataset, self).__init__()
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user = torch.tensor(row[0], dtype=torch.long)
        item = torch.tensor(row[1], dtype=torch.long)
        label = torch.tensor(row[2], dtype=torch.float)

        return user, item, label
