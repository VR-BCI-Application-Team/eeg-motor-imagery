import os
import numpy as np
from typing import Optional

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class EGGDataset(Dataset):
    def __init__(self, data_dir: str, is_train: bool = True, transform = None):
        super(EGGDataset, self).__init__()
        self.is_train = is_train
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        if is_train:
            data_dir = os.path.join(data_dir, 'train')
        else:
            data_dir = os.path.join(data_dir, 'eval')

        self.data_path = data_dir
        self.data_list = []

        self.data_list += [os.path.join('left', x) for x in os.listdir(os.path.join(data_dir, 'left'))]
        self.data_list += [os.path.join('right', x) for x in os.listdir(os.path.join(data_dir, 'right'))]
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        input = np.load(os.path.join(self.data_path, file_name))

        if 'left' in file_name:
            label = torch.tensor(0)  # Left=0, Right=1
        else:
            label = torch.tensor(1)

        input = self.transform(input)

        return input, label


class EGGDataloader(pl.LightningDataModule):
    def __init__(self, data_dir: str = './dataset', batch_size: int = 32, transform = None):
        super(EGGDataloader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self,):
        # TODO: download
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.egg_train_set = EGGDataloader(self.data_dir, transform=self.transform)
        self.egg_eval_set = EGGDataloader(self.data_dir, is_train=False, transform=self.transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.egg_train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.egg_eval_set, batch_size=self.batch_size)
