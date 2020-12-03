# pylint: disable=invalid-name
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import time

import torch

import experiments

from model import Model
from trainer import Trainer
from tester import Tester
from zsamples import ZSamples
from datasets import DataSets
from plotter import Plotter
from writer import Writer


# %%

experiment = experiments.EXPERIMENT_1

BATCH_SIZE = 2048

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Create the z-samples set.
z_samples = ZSamples(experiment=experiment, device=device,)

# Create the trainable model.
model = Model(
    z_samples.Z_SPACE_SIZE,
    len(datasets.x_dimension_names),
    device=device,
    # hidden_size=1536,  # exp 6
).to(device=device)

# %%

# Create the trainer.
trainer = Trainer(
    experiment=experiment, z_samples=z_samples, model=model, device=device,
)

# Create the plotter, this object will render all plots to the notebook.
plotter = Plotter(
    datasets=datasets,
    z_samples=z_samples,
    test_s=0.9,
    train_s=0.2,
    zline_s=5,
    zline_skip=datasets.x_test.shape[0] // 50,
)

# Create the writer, this object will write information that can be browsed using tensorboard.
writer = Writer(
    datasets_target_function=datasets.target_function_desc,
    trainer_params=trainer.params_desc,
    datasets_params=datasets.params_desc,
)

# Create the tester, this object will run goal 1, goal 2 and emd tests.
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

# Iterate running the training algorithm.
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
