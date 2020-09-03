"""
This module can be used to compose a stochastic function and train a neural network
to aproximate it.
"""
# pylint: disable=invalid-name

import time

import torch

from model import StochasticFFNN
from trainer import Trainer
from tester import Tester
from zsamples import ZSamples
from datasets import DataSets
from plotter_windowed import PlotterWindowed

from experiments import (
    EXPERIMENT_1,
    EXPERIMENT_2,
    EXPERIMENT_3,
    EXPERIMENT_4,
    EXPERIMENT_5,
    EXPERIMENT_6,
)


def main():
    """
    Instantiate the objects and run the training loop.
    """
    # Use coprime numbers to prevent any matching points between train and test.
    TRAIN_SIZE = 31013
    # TRAIN_SIZE = 9611

    TEST_SIZE = 5007
    # TEST_SIZE = 1001

    experiment = EXPERIMENT_1

    BATCH_SIZE = 2048

    device = torch.device("cuda")

    if "dataset_builder" not in experiment:
        datasets = DataSets.generated_dataset(
            base_function=experiment["base_function"],
            noise_function=experiment["noise_function"],
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
            x_range_train=experiment["x_range_train"],
            x_range_test=experiment["x_range_test"],
            batch_size=BATCH_SIZE,
            device=device,
        )
    else:
        datasets = experiment["dataset_builder"](BATCH_SIZE, device)
    # datasets.show()

    z_samples = ZSamples(
        z_samples_per_dimension=experiment["z_samples_per_dimension"],
        z_range=experiment["z_range"],
        outer_level_scalar=experiment["outer_level_scalar"],
        device=device,
    )

    stochastic_ffnn = StochasticFFNN(
        z_samples.Z_SPACE_SIZE,
        len(datasets.x_dimensions),
        device=device,
        # hidden_size=1536,  # exp 6
    ).to(device=device)

    trainer = Trainer(
        z_samples=z_samples,
        movement=experiment["movement"],
        model=stochastic_ffnn,
        learning_rate=experiment["learning_rate"],
        milestones=[60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660,],
        gamma=experiment["gamma"],
        device=device,
    )

    plotter = PlotterWindowed(
        datasets=datasets,
        z_samples_size=z_samples.z_samples.shape[0],
        test_s=0.9,
        train_s=0.2,
        zline_s=5,
        zline_skip=TEST_SIZE // 50,
        zsample_label=True,
    )

    tester = Tester(
        z_samples=z_samples,
        datasets=datasets,
        trainer=trainer,
        plotter=plotter,
        model=stochastic_ffnn,
        skip_epochs=experiment["skip_epochs"],
        device=device,
    )

    for epoch in range(0, experiment["num_epochs"]):
        start = time.time()

        for x, y in datasets.data_loader_train:
            trainer.batch(
                x_pt=x, y_pt=y,
            )

        trainer.step(epoch=epoch)

        tester.step(epoch=epoch)

        end = time.time()
        print("epoch: {} elapsed time: {}".format(epoch, end - start))


if __name__ == "__main__":
    main()
