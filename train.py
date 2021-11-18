from datetime import datetime
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import EGGDataloader
from model import Classifier


def args():
    parser = ArgumentParser()
    parser.add_argument()
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    opt = args()
    now = datetime.now()
    wandb_logger = WandbLogger(
        project='EGG_Motor_Imagery',
        experiment=f'exp_{now.strftime("%H:%M_%d/%m")}',
    )
    wandb_logger.experiment.config.update(opt)

    # get data
    dataset = EGGDataloader()
    dataset.setup()

    # model + trainer
    model = Classifier()
    wandb_logger.experiment.watch(model)

    trainer = pl.Trainer.from_argparse_args(
        args=opt,
        logger=wandb_logger
    )

    trainer.fit(model, dataset)


if __name__ == '__main__':
    main()
