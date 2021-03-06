"""
This module contains class DataSets.
"""
# pylint: disable=bad-continuation

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.model_selection import train_test_split

from utils import sample_uniform, to_tensor


class FunctionDataSet(Dataset):
    """
    This class implements a dataset by wrapping x and y data arrays.
    """

    def __init__(self, x_np, y_np, device):
        self.length = x_np.shape[0]

        # Convert numpy arrays to pytorch tensors.
        self.x_data = to_tensor(x_np, device)
        self.y_data = to_tensor(y_np, device)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


class DataSets:
    """
    This class creates train and test datasets from the provided dimensions and functions.
    """

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        x_dimension_names,
        y_dimension_name,
        batch_size,
        target_function_desc,
        params_desc,
        device,
    ):
        """
        This function initializes the training and testing data sets.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.x_dimension_names = x_dimension_names
        self.y_dimension_name = y_dimension_name

        self.target_function_desc = target_function_desc
        self.params_desc = params_desc

        dataset_train = FunctionDataSet(x_np=x_train, y_np=y_train, device=device)

        self.data_loader_train = DataLoader(
            dataset=dataset_train, batch_size=batch_size, shuffle=True
        )

    @staticmethod
    def generated_dataset(
        experiment, train_size, test_size, batch_size, device,
    ):
        """
        This named constructor builds a DataSet from a pair of base and noise functions.
        """
        base_function = experiment["base_function"]
        noise_function = experiment["noise_function"]
        x_range_train = experiment["x_range_train"]
        x_range_test = experiment["x_range_test"]

        target_function_desc = "{}/{}".format(base_function.name, noise_function.name)
        target_function = lambda x, y: noise_function(*base_function(x, y))

        params_desc = "train size: {}/test size: {}".format(train_size, test_size)

        def create_dataset(function, size, x_ranges):
            """
            This function takes a composed function a size and a range and
            then creates an artificial dataset based on the function.
            """
            # x_np = sample_random(x_ranges, size, base_function.x_space_size)
            x_np = sample_uniform(x_ranges, size, base_function.x_space_size)
            y_np = np.zeros((size, base_function.y_space_size))
            x_np, y_np = function(x_np, y_np)

            return x_np, y_np

        # Create the training dataset.
        x_train, y_train = create_dataset(target_function, train_size, x_range_train)

        # Create the test dataset.
        x_test, y_test = create_dataset(target_function, test_size, x_range_test)

        return DataSets(
            x_train,
            y_train,
            x_test,
            y_test,
            [str(i) for i in range(base_function.x_space_size)],
            "Y",
            batch_size,
            target_function_desc,
            params_desc,
            device,
        )

    @staticmethod
    def california_housing_dataset(batch_size, device):
        """
        This named constructor builds a DataSet from the California Housing dataset.
        """
        from sklearn.datasets import fetch_california_housing

        cal_housing = fetch_california_housing()

        non_outliers = np.ones(cal_housing.data.shape[0], dtype=bool)
        for idx in range(cal_housing.data.shape[1]):
            column = cal_housing.data[:, idx]
            cutoffs = np.percentile(column, (1.0, 99.0))
            non_outliers = np.logical_and(
                non_outliers, np.logical_and(column > cutoffs[0], column < cutoffs[1])
            )
        # cutoffs = np.percentile(cal_housing.target, (1.0, 99.0))
        # non_outliers = np.logical_and(
        #    non_outliers, np.logical_and(cal_housing.target > cutoffs[0],
        #    cal_housing.target < cutoffs[1])
        # )

        cal_housing.data = cal_housing.data[non_outliers]
        cal_housing.target = cal_housing.target[non_outliers]

        x_train, x_test, y_train, y_test = train_test_split(
            cal_housing.data, cal_housing.target, test_size=0.2, random_state=0
        )

        non_outliers = np.ones(x_test.shape[0], dtype=bool)
        for idx in range(x_test.shape[1]):
            column = x_test[:, idx]
            cutoffs = np.percentile(column, (5.0, 95.0))
            non_outliers = np.logical_and(
                non_outliers, np.logical_and(column > cutoffs[0], column < cutoffs[1])
            )
        x_test = x_test[non_outliers]
        y_test = y_test[non_outliers]

        x_dimensions = cal_housing.feature_names

        train_size = x_train.shape[0]
        test_size = x_test.shape[0]

        y_train = y_train[..., np.newaxis]
        y_test = y_test[..., np.newaxis]

        params_desc = "train size: {}/test size: {}".format(train_size, test_size)

        return DataSets(
            x_train,
            y_train,
            x_test,
            y_test,
            x_dimensions,
            "Price",
            batch_size,
            "california housing dataset",
            params_desc,
            device,
        )

    @staticmethod
    def load_csv(batch_size, device, data_file):
        """
        This named constructor builds a Dataset from a time series.
        """
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        pandas_dataframe = pd.read_csv(data_file)

        pandas_dataframe = pandas_dataframe.dropna(
            how="any", subset=["DEPARTURE_DELAY", "ARRIVAL_DELAY"]
        )
        column = "DEPARTURE_DELAY"
        q_hi = pandas_dataframe[column].quantile(0.95)

        pandas_dataframe = pandas_dataframe[
            (pandas_dataframe[column] < q_hi)  # & (pandas_dataframe[column] > q_low)
        ]
        training_set = pandas_dataframe.loc[
            :, ["DEPARTURE_DELAY", "ARRIVAL_DELAY"]
        ].values

        x_train, x_test, y_train, y_test = train_test_split(
            training_set[:, 0], training_set[:, 1], test_size=0.3, random_state=0,
        )

        params_desc = "train size: {}/test size: {}".format(
            x_train.shape[0], x_test.shape[0]
        )

        return DataSets(
            x_train[:, np.newaxis],
            y_train[:, np.newaxis],
            x_test[:, np.newaxis],
            y_test[:, np.newaxis],
            ["departure delay"],
            "arrival delay",
            batch_size,
            "Flight delays",
            params_desc,
            device,
        )
