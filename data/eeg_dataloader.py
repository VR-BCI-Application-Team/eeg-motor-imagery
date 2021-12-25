from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class EGGDataset(Dataset):
    def __init__(self, data_dir: str, is_train: bool = True, transform = None):
        super(EGGDataset, self).__init__()
        self.is_train = is_train
        self.transform = transform
        
        

    def __len__(self):
        pass

    def __getitem__(self, idx):
        input = None
        label = None

        input = self.transform(input)

        return input, label


class EGGDataloader(pl.LightningDataModule):
    def __init__(self, data_dir: str = './dataset', batch_size: int = 32, transform = None):
        super(EGGDataloader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage: Optional[str] = None) -> None:
        self.egg_train_set = EGGDataloader(self.data_dir, is_train=True, transform=self.transform)
        self.egg_eval_set = EGGDataloader(self.data_dir, is_train=False, transform=self.transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.egg_train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.egg_eval_set, batch_size=self.batch_size)
