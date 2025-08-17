#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import ClusteringMetric
from sklearn.cluster import KMeans
from mealpy import *


class KMeansParametersProblem(Problem):
    def __init__(self, bounds=None, minmax="min", X=None, obj_name=None, seed=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.X = X
        self.obj_name = obj_name
        self.seed = seed

    def get_model(self, solution) -> KMeans:
        x_dict = self.decode_solution(solution)
        kmeans = KMeans(random_state=self.seed, n_init="auto")
        kmeans.set_params(**x_dict)
        kmeans.fit(self.X)
        return kmeans

    def obj_func(self, solution):
        kmeans = self.get_model(solution)
        y_pred = kmeans.predict(self.X)
        evaluator = ClusteringMetric(y_pred=y_pred, X=self.X, raise_error=False, decimal=8)
        obj = evaluator.get_metric_by_name(self.obj_name)[self.obj_name]
        return obj


class KCentersClusteringProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, obj_name=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.data = data
        self.obj_name = obj_name

    @staticmethod
    def get_y_pred(X, solution):
        centers = np.reshape(solution, (-1, X.shape[1]))
        # Calculate the distance between each sample and each center
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2))
        # Assign each sample to the closest center
        labels = np.argmin(distances, axis=1)
        return labels

    def get_metrics(self, solution=None, list_metric=None, list_paras=None):
        centers = np.reshape(solution, (-1, self.data.X.shape[1]))
        y_pred = self.get_y_pred(self.data.X, centers)
        evaluator = ClusteringMetric(y_true=self.data.y, y_pred=y_pred, X=self.data.X, decimal=8)
        results = evaluator.get_metrics_by_list_names(list_metric, list_paras)
        return results

    def obj_func(self, solution):
        centers = self.decode_solution(solution)["center_weights"]
        y_pred = self.get_y_pred(self.data.X, centers)
        evaluator = ClusteringMetric(y_true=self.data.y, y_pred=y_pred, X=self.data.X, raise_error=False, decimal=8)
        obj = evaluator.get_metric_by_name(self.obj_name)[self.obj_name]
        return obj
