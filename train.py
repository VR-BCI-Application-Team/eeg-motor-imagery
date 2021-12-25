from datetime import datetime
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import EGGDataloader
from model import Classifier
from model.callbacks import WandbSaveWeight



def args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()


def main():
    opt = args()
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(opt)
    checkpoint_callbacks = ModelCheckpoint(
        monitor='Valid/loss',
        dirpath=f'exp',
        save_weights_only=True,
        save_top_k=1,
        filename='weight'
    )
    wandb_callbacks = WandbSaveWeight()

    # get data
    dataset = EGGDataloader(data_dir='./data/bci_2a')
    # dataset.setup()

    # model + trainer
    model = Classifier(n_channels=25, learning_rate=opt.lr)
    wandb_logger.experiment.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args=opt,
        # auto_lr_find=True,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callbacks, wandb_callbacks]
    )

    # lr_finder = trainer.tune(model)
    
    # wandb_logger.experiment.log({"lr_finder": wandb.Image(lr_finder.plot(suggest=True))})
    # lr_finder.results

    # model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model, dataset)


if __name__ == '__main__':
    main()
