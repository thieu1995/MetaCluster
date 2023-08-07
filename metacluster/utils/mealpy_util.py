#!/usr/bin/env python
# Created by "Thieu" at 14:52, 26/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy import *
from permetrics import ClusteringMetric


class KCenterClusteringProblem(Problem):
    def __init__(self, lb, ub, minmax, data=None, estimator=None, obj_name=None, obj_paras=None,
                 name="K Center Clustering Problem", **kwargs):
        ## data is assigned first because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.estimator = estimator
        self.obj_name = obj_name
        self.obj_paras = obj_paras
        self.name = name
        super().__init__(lb, ub, minmax, **kwargs)

    def get_y_pred(self, X, solution):
        centers = np.reshape(solution, (-1, X.shape[1]))
        # Calculate the distance between each sample and each center
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2))
        # Assign each sample to the closest center
        labels = np.argmin(distances, axis=1)
        return labels

    def get_metrics(self, solution=None, list_metric=None, list_paras=None):
        centers = np.reshape(solution, (-1, self.data.X.shape[1]))
        y_pred = self.get_y_pred(self.data.X, centers)
        evaluator = ClusteringMetric(y_true=self.data.y, y_pred=y_pred, X=self.data.X)
        results = evaluator.get_metrics_by_list_names(list_metric, list_paras)
        return results

    # def amend_position(self, position=None, lb=None, ub=None):
    #     n_features = self.data.X.shape[1]
    #     n_clusters = int(len(position) / n_features)
    #     pos = np.clip(position, lb, ub)
    #     centers = np.reshape(pos, (n_clusters, n_features))
    #     y_pred = self.get_y_pred(self.data.X, centers)
    #     while len(np.unique(y_pred, return_counts=True)[0]) == 1:
    #         centers[np.random.randint(0, n_clusters)] = np.random.uniform(lb[:n_features], ub[:n_features])
    #         y_pred = self.get_y_pred(self.data.X, centers)
    #     return centers.flatten()

    def fit_func(self, solution):
        centers = np.reshape(solution, (-1, self.data.X.shape[1]))
        y_pred = self.get_y_pred(self.data.X, centers)
        evaluator = ClusteringMetric(y_true=self.data.y, y_pred=y_pred, X=self.data.X, raise_error=False)
        obj = evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name]
        return obj
