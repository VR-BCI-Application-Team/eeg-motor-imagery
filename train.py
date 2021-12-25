from datetime import datetime
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import EGGDataloader
from model import Classifier


def args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    return parser.parse_args()


def main():
    opt = args()
    now = datetime.now()
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(opt)

    # get data
    dataset = EGGDataloader(data_dir='./data/bci_2a')
    dataset.setup()

    # model + trainer
    model = Classifier()
    wandb_logger.experiment.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args=opt,
        logger=wandb_logger,
    )

    trainer.fit(model, dataset)


if __name__ == '__main__':
    main()
