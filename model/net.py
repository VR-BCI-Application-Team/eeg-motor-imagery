import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
  

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
    def __init__(self) -> None:  
        super().__init__()
        self.backbone = EGGNet()
        self.accuracy = torchmetrics.Accuracy()
        # self.auc = torchmetrics.AUROC()
        # self.recall = torchmetrics.Recall()
        # self.precision = torchmetrics.Precision()


    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.backbone(x.float())
        loss = nn.BCELoss(preds.squeeze().float(), y.float())

        # metrics
        self.accuracy(preds, y)  # same shape, or 2D vs 1D

        # logging
        self.log('Train/loss', loss, on_epoch=True)
        self.log('Train/accuracy', self.accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.backbone(x.float())
        loss = nn.BCELoss(preds.squeeze().float(), y.float())

        # metrics
        self.accuracy(preds, y)

        # logging
        self.log('Valid/loss', loss, on_step=True)
        self.log('Valid/accuracy', self.accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
