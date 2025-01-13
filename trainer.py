import datetime
import os

import lightning as L
from lightning.pytorch import loggers

def add_arguments(parser):
    # Use this argument to determine whether to train on CPU or GPU.
    parser.add_argument('--trainer_accelerator', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--trainer_max_epochs', type=int, default=1000)
    parser.add_argument('--trainer_seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--trainer_check_val_every_n_epoch', type=int, default=10)

def fit_model(args, model, dataset):
    L.seed_everything(args.trainer_seed)
    # By default, we will save each experiment in a folder specifying the current time.
    save_dir = args.save_dir or os.path.join('data', datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
    logger = loggers.CSVLogger(save_dir=save_dir)
    # L.Trainer has many more arguments, so if you want to change anything here, simply look up the L.Trainer class!
    trainer = L.Trainer(
        accelerator=args.trainer_accelerator,
        logger=logger,
        max_epochs=args.trainer_max_epochs,
        check_val_every_n_epoch=args.trainer_check_val_every_n_epoch
    )
    trainer.fit(model, dataset)
