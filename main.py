"""
This module can be used to compose a stochastic function and train a neural network
to aproximate it.
"""
# pylint: disable=invalid-name
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import time

import torch

import experiments

from model import StochasticFFNN
from trainer import Trainer
from tester import Tester
from zsamples import ZSamples
from datasets import DataSets
from plotter import Plotter
from writer import Writer


# %%

experiment = experiments.EXPERIMENT_7

BATCH_SIZE = 128
# BATCH_SIZE = 2048

device = torch.device("cuda")

# If using a generated dataset.
if "dataset_builder" not in experiment:
    # Use coprime numbers to prevent any matching points between train and test.
    TRAIN_SIZE = 31013
    # TRAIN_SIZE = 9611

    TEST_SIZE = 5007
    # TEST_SIZE = 1001

    datasets = DataSets.generated_dataset(
        experiment=experiment,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
    )
# If using a real data dataset.
else:
    datasets = experiment["dataset_builder"](BATCH_SIZE, device)
# datasets.show()

z_samples = ZSamples(experiment=experiment, device=device,)

model = StochasticFFNN(
    output_size=datasets.output_size,
    x_space_size=len(datasets.x_dimensions),
    device=device,
    # hidden_size=1536,  # exp 6
).to(device=device)

# %%

trainer = Trainer(
    experiment=experiment, z_samples=z_samples, model=model, device=device,
)

plotter = Plotter(
    datasets=datasets,
    z_samples=z_samples,
    test_s=0.9,
    train_s=0.2,
    # zline_s=2,
    # zline_skip=datasets.x_test.shape[0] // 600,
    zline_s=5,
    zline_skip=datasets.x_test.shape[0] // 50,
)

writer = Writer(
    datasets_target_function=datasets.target_function_desc,
    trainer_params=trainer.params_desc,
    datasets_params=datasets.params_desc,
)

tester = Tester(
    experiment=experiment,
    z_samples=z_samples,
    datasets=datasets,
    plotter=plotter,
    writer=writer,
    model=model,
    device=device,
)

# %%

for epoch in range(0, experiment["num_epochs"]):
    start = time.time()

    for x, y in datasets.data_loader_train:
        trainer.batch(
            x_pt=x, y_pt=y,
        )

    trainer.step(epoch=epoch)

    tester.step(epoch=epoch)

    end = time.time()
    print(f"epoch: {epoch} elapsed time: {end - start}")

# %%
