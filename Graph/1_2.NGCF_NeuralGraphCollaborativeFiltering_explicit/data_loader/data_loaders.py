import os
import pandas as pd

from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NGCF_explicit_DataLoader(BaseDataLoader):
    def __init__(self, data_dir, file_name, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        origin_df = pd.read_csv(os.path.join(data_dir, file_name))

        self.data_dir = data_dir

        super().__init__(self.dataset, file_name, batch_size,
                         shuffle, validation_split, num_workers)
