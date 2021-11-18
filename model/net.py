import torch
import torch.nn as nn

import pytorch_lightning as pl


class EGGNet(nn.Module):
    def __init__(self):
        super(EGGNet, self).__init__()

    def forward(self, x):
        pass

    def frozen_until(self, to_layer):
        pass


class Classifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = EGGNet()
        # self.accuracy 

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        # loss = F.cross_entropy(y_hat, y)
        # self.log('Train/loss', loss, on_epoch=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        # loss = F.cross_entropy(y_hat, y)
        # self.log('Valid/loss', loss, on_step=True)
        # self.log('Valid/accuracy', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        # loss = F.cross_entropy(y_hat, y)
        # self.log('Test/loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
