from dataclasses import MISSING, dataclass, field
from typing import Union, List, Any
import argparse

import models
import datasets
import trainer

def get_parser():
    # This parser contains all the arguments that your code will be looking for.
    # You can add different arguments in the parts of your code that specify the model, datasets and trainer.
    # We will follow the convention that all model arguments start with 'model_' (and the same goes for
    # dataset arguments and trainer arguments).
    parser = argparse.ArgumentParser()
    models.add_arguments(parser)
    datasets.add_arguments(parser)
    trainer.add_arguments(parser)
    return parser

def main(args):
    # This returns the model, as a LightningModule. This allows us to use Pytorch Lightning.
    model = models.get_model(args)
    # This returns a dataloader. Later, we will extend this to also allow for validation datasets.
    dataset = datasets.get_dataset(args)
    # This function fits the model.
    trainer.fit_model(args, model, dataset)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
