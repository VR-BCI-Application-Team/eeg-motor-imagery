import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl

import wandb
  

class EGGNet(nn.Module):
    def __init__(self, n_channels=64, timestamp=120):
        # model from: https://github.com/aliasvishnu/EEGNet/
        super(EGGNet, self).__init__()
        # Layer 1
        self.T = timestamp
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(1, 16, (1, n_channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        self.fc1 = nn.Linear(4*2*8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x.shape[-2:] == (self.n_channels, 126), (
            f"Miss match input shape, got ({x.shape})")
        # Permute [x, y, 25, 126] -> [x, y, 126, 25]
        x = x.permute(0, 1, 3, 2)

        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.contiguous().view(-1, 4*2*8)
        x = self.sigmoid(self.fc1(x))
        return x

    def frozen_until(self, to_layer):
        pass


class Classifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, n_channels=25) -> None:  
        super().__init__()
        self.backbone = EGGNet(n_channels=n_channels)
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.train_F1acc = torchmetrics.F1(num_classes=2, multiclass=True)
        self.valid_F1acc = torchmetrics.F1(num_classes=2, multiclass=True)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.backbone(x.float()).squeeze()
        loss = F.binary_cross_entropy(preds.float(), y.float())

        # metrics
        self.train_acc(preds, y.int())
        self.train_F1acc(preds, y.int())

        # logging
        self.log('Train/loss', loss, on_epoch=True)
        self.log('Train/accuracy_step', self.train_acc)
        self.log('Train/F1_step', self.train_F1acc)
        
        return {"loss": loss, "preds": preds.detach(), "targets": y}

    def training_epoch_end(self, outputs):
        self.log('Train/accuracy_epoch', self.train_acc)

        mean_F1 = self.train_F1acc.compute()
        self.log('Train/F1_epoch', mean_F1)
        self.train_F1acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.backbone(x.float()).squeeze()
        loss = F.binary_cross_entropy(preds.float(), y.float())

        # metrics
        self.valid_acc(preds, y.int())
        self.valid_F1acc(preds, y.int())

        # logging
        self.log('Valid/loss', loss, on_epoch=True)
        self.log('Valid/accuracy_step', self.valid_acc)
        self.log('Valid/F1_step', self.valid_F1acc)

        return {"loss":loss, "preds": preds.detach(), "targets": y}

    def validation_epoch_end(self, outs):
        self.log('Valid/accuracy_epoch', self.valid_acc)

        mean_F1 = self.valid_F1acc.compute()
        self.log('Valid/F1_epoch', mean_F1)
        self.valid_F1acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
