## A simple template for running neural network experiments

This repository contains a simple repository for running neural network experiments. It is split up into four files:

- `train.py` is the script you will execute.
- `models.py` contains all the models you will create. The function `get_model(args)` should depend on the command-line arguments and return the model you want to train. You can add new command line arguments by modifying the function `models.add_arguments`.
- `datasets.py` will create the dataset, using the function `get_dataset(args)`. Again, you can add new command line arguments by modifying the function `dataset.add_arguments`. Note that right now, you should only create a training dataloader. We will talk about validation dataloaders next week.
- `trainer.py` contains the infrastructure that allows Pytorch Lightning to train the model. Ultimately, we will use `trainer.fit_model` to train the model.
