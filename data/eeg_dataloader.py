from typing import Optional

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class EGGDataset(Dataset):
    def __init__(self):
        super(EGGDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class EGGDataloader(pl.LightningDataModule):
    def __init__(self):
        super(EGGDataloader, self).__init__()
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.batch_size = 32

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.batch_size)
