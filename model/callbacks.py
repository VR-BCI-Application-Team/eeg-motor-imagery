import os
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WandbSaveWeight(Callback):
    def __init__(self) -> None:
        super().__init__()
        
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        weight_path = "./exp/weight.ckpt"
        wandb_experiment = trainer.logger.experiment
        weights = wandb.Artifact("Pre-trained", type="WEIGHTS")
        weights.add_file(weight_path)
        wandb_experiment.log_artifact(weights)
        os.remove(weight_path)

