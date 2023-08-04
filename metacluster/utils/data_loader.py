#!/usr/bin/env python
# Created by "Thieu" at 23:33, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray, Optional, default=None
        The labels of your data, for clustering problem, this can be None
    """

    SUPPORT = {
        "scaler": ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer"]
    }

    def __init__(self, X, y=None, name="Unknown"):
        self.X = X
        self.y = y
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        if self.y is None:
            self.X_train, self.X_test = train_test_split(self.X, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    @staticmethod
    def scale(X, method="MinMaxScaler", **kwargs):
        if method in Data.SUPPORT["scaler"]:
            scaler = getattr(preprocessing, method)(**kwargs)
            data = scaler.fit_transform(X)
            return data, scaler
        raise ValueError(f"Data class doesn't support scaling method name: {method}")

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_name(self):
        return self.name


def get_dataset(dataset_name):
    """
    Helper function to retrieve the data

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    data: Data
        The instance of Data class, that hold X and y variables.
    """
    dir_root = f"{Path(__file__).parent.parent.__str__()}/data"
    list_path = Path(f"{dir_root}").glob("*.csv")
    list_datasets = [pf.name[:-4] for pf in list_path]

    if dataset_name not in list_datasets:
        print(f"MetaCluster currently does not have '{dataset_name}' data in its database....")
        print("+ List of the supported datasets are:")
        for idx, dataset in enumerate(list_datasets):
            print(f"\t{idx + 1}: {dataset}")
    else:
        df = pd.read_csv(f"{dir_root}/{dataset_name}.csv", header=None)
        data = Data(np.array(df.iloc[:, 0:-1]), np.array(df.iloc[:, -1]), name=dataset_name)
        print(f"Requested dataset: {dataset_name} found and loaded!")
        return data
