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

# Use coprime numbers to prevent any matching points between train and test.
TRAIN_SIZE = 31013
# TRAIN_SIZE = 9611

TEST_SIZE = 5007
# TEST_SIZE = 1001


experiment = experiments.EXPERIMENT_1a

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

z_samples = ZSamples(experiment=experiment, device=device,)

model = StochasticFFNN(
    z_samples.Z_SPACE_SIZE,
    len(datasets.x_dimensions),
    device=device,
    # hidden_size=1536,  # exp 6
).to(device=device)

# %%

trainer = Trainer(
    experiment=experiment, z_samples=z_samples, model=model, device=device,
)

plotter = Plotter(
    datasets=datasets,
    z_samples_size=z_samples.z_samples.shape[0],
    test_s=0.9,
    train_s=0.2,
    zline_s=5,
    zline_skip=TEST_SIZE // 50,
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
