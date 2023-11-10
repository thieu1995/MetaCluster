#!/usr/bin/env python
# Created by "Thieu" at 09:02, 10/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Optional
import numpy as np
from metacluster.utils import mealpy_util as mu, cluster, validator


class MhaKMeansTuner:
    """
    Defines a MhaKMeansTunerTuner class that can optimize all hyper-parameters of KMeans model (Dataset doesn't need labels)

    Parameters
    ----------
    optimizer: str or mu.Optimizer, default = None
        The string that represent class optimizer or an instance of Optimizer class from Mealpy library.
        Current supported optimizers, please check it here: https://github.com/thieu1995/mealpy
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.
        Please use this to get supported optimizers: MhaKMeansTuner.get_support(name="optimizer")

    optimizer_paras: dict, default=None
        A dictionary that present the parameters of Optimizer class.
        You can set it to None to use all default parameters in Mealpy library.

    seed : int, default=20
        Determines random number generation for the whole program. Use an int to make the randomness deterministic.

    Examples
    --------
    The following example shows how to use  the most informative features in the MhaKMeansTuner

    >>> from metacluster import get_dataset, MhaKMeansTuner
    >>> import time
    >>>
    >>> data = get_dataset("aggregation")
    >>> data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))
    >>>
    >>> # Get all supported methods and print them out
    >>> MhaKMeansTuner.get_support(name="all")
    >>>
    >>> time_run = time.perf_counter()
    >>> model = MhaKMeansTuner(optimizer="OriginalWOA", optimizer_paras={"name": "GWO", "epoch": 10, "pop_size": 20}, seed=10)
    >>> model.fit(data.X, mealpy_bound=None, max_clusters=5, obj="SSEI", verbose=True)
    >>>
    >>> print(model.best_parameters)
    >>> print(model.best_estimator.predict(data.X))
    >>> print(model.predict(data.X))
    >>>
    >>> print(f"Time process: {time.perf_counter() - time_run} seconds")
    """

    SUPPORT = {
        "obj": cluster.get_all_clustering_metrics(),
        "metrics": cluster.get_all_clustering_metrics(),
        "optimizer": list(mu.get_all_optimizers().keys())
    }

    def __init__(self, optimizer=None, optimizer_paras=None, seed=20):
        self.seed = seed
        self.optimizer, self.optimizer_paras = self._set_optimizer(optimizer, optimizer_paras)
        self.best_parameters = {}
        self.best_estimator: Optional[mu.KMeans] = None

    @staticmethod
    def get_support(name="all", verbose=True):
        if name == "all":
            if verbose:
                for key, value in MhaKMeansTuner.SUPPORT.items():
                    print(f"Supported methods for '{key}' are: ")
                    print(value)
            return MhaKMeansTuner.SUPPORT
        if name in list(MhaKMeansTuner.SUPPORT.keys()):
            if verbose:
                print(f"Supported methods for '{name}' are: ")
                print(MhaKMeansTuner.SUPPORT[name])
            return MhaKMeansTuner.SUPPORT[name]
        raise ValueError(f"MhaKMeansTuner doesn't support {name}.")

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if optimizer_paras is None:
            optimizer_paras = {}
        if type(optimizer) is str:
            opt_class = mu.get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                optimizer = opt_class(**optimizer_paras)
            else:
                optimizer = opt_class(epoch=100, pop_size=20)
        elif isinstance(optimizer, mu.Optimizer):
            if type(optimizer_paras) is dict:
                optimizer.set_parameters(optimizer_paras)
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")
        return optimizer, optimizer_paras

    def _set_obj(self, obj=None):
        if obj in list(self.SUPPORT["obj"].keys()):
            if obj[-1] == "S":
                raise ValueError(f"Invalid obj. MhaKMeansTuner only supports internal objective.")
            return obj
        else:
            print(f"Invalid obj. MhaKMeansTuner doesn't support obj: {obj}")

    def fit(self, X, mealpy_bound=None, max_clusters=10, obj="SSEI", verbose=True, mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        X : np.ndarray,
            The matrix feature X of dataset. Make sure your matrix X is normalized or standardized

        mealpy_bound : Union[List, mu.BaseVar], default=None.
            It is like a param grid search for you KMeans model, but you need to define it using DataType from mealpy: https://github.com/thieu1995/mealpy#examples
            If it is `None`, 4 hyper-parameters will be tuned include: `n_clusters`, `init`, `max_iter`, and `algorithm`.
            Please check all hyper-parameters from here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        max_clusters : int, default=10.
            The maximum of clusters you want to search for this specific dataset X

        obj : str, default=""SSEI""
            A string that represent objective name that supported by the library: https://github.com/thieu1995/permetrics
            To get the supported metrics, please use: MhaKCentersClustering.get_support(), supported obj are supported metrics

        verbose : int, default = True
            Controls verbosity of output for each training process of each optimizer.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) used in Optimizer (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.obj = self._set_obj(obj)
        minmax = self.SUPPORT["obj"][self.obj]
        log_to = "console" if verbose else "None"
        if mealpy_bound is None:
            mealpy_bound = [
                mu.IntegerVar(lb=2, ub=max_clusters, name="n_clusters"),
                mu.StringVar(valid_sets=('k-means++', 'random'), name="init"),
                mu.MixedSetVar(valid_sets=(200, 250, 300, 350, 400, 450, 500), name="max_iter"),
                mu.StringVar(valid_sets=("lloyd", "elkan"), name="algorithm")
            ]
        problem = mu.KMeansParametersProblem(bounds=mealpy_bound, minmax=minmax, X=X, obj_name=self.obj, log_to=log_to, seed=self.seed)
        self.optimizer.solve(problem, mode=mode, n_workers=n_workers, termination=termination, seed=self.seed)
        self.best_parameters = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)
        self.best_estimator = self.optimizer.problem.get_model(self.optimizer.g_best.solution)
        return self
    
    def predict(self, X):
        if self.best_estimator is None:
            raise ValueError(f"The model is not trained yet. You need to call fit() function to train the model.")
        return self.best_estimator.predict(X)


class MhaKCentersClustering:
    """
    Defines a MhaKCentersClustering class that can optimize K-Centers Clustering Problem (Dataset need labels)

    Parameters
    ----------
    optimizer: str or mu.Optimizer, default = None
        The string that represent class optimizer or an instance of Optimizer class from Mealpy library.
        Current supported optimizers, please check it here: https://github.com/thieu1995/mealpy
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.
        Please use this to get supported optimizers: MhaKMeansTuner.get_support(name="optimizer")

    optimizer_paras: dict, default=None
        A dictionary that present the parameters of Optimizer class.
        You can set it to None to use all default parameters in Mealpy library.

    seed : int, default=20
        Determines random number generation for the whole program. Use an int to make the randomness deterministic.

    Examples
    --------
    The following example shows how to use  the most informative features in the MhaKCentersClustering

    >>> from metacluster import get_dataset, MhaKCentersClustering
    >>> import time
    >>>
    >>> data = get_dataset("aggregation")
    >>> data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))
    >>>
    >>> # Get all supported methods and print them out
    >>> MhaKCentersClustering.get_support(name="all")
    >>>
    >>> time_run = time.perf_counter()
    >>> model = MhaKCentersClustering(optimizer="OriginalWOA", optimizer_paras={"name": "GWO", "epoch": 10, "pop_size": 20}, seed=10)
    >>> model.fit(data, cluster_finder="elbow", obj="SSEI", verbose=True)
    >>>
    >>> print(model.best_agent)
    >>> print(model.predict(data.X))
    >>>
    >>> print(f"Time process: {time.perf_counter() - time_run} seconds")
    """

    SUPPORT = {
        "cluster_finder": {"elbow": "get_clusters_by_elbow", "gap": "get_clusters_by_gap_statistic",
                           "silhouette": "get_clusters_by_silhouette_score", "davies_bouldin": "get_clusters_by_davies_bouldin",
                           "calinski_harabasz": "get_clusters_by_calinski_harabasz", "bayesian_ìnormation": "get_clusters_by_bic",
                           "all_min": "get_clusters_all_min", "all_max": "get_clusters_all_max",
                           "all_mean": "get_clusters_all_mean", "all_majority": "get_clusters_all_majority"},
        "obj": cluster.get_all_clustering_metrics(),
        "metrics": cluster.get_all_clustering_metrics(),
        "optimizer": list(mu.get_all_optimizers().keys())
    }
    def __init__(self, optimizer=None, optimizer_paras=None, seed=20):
        self.seed = seed
        self.optimizer, self.optimizer_paras = self._set_optimizer(optimizer, optimizer_paras)
        self.best_agent = None
        self.convergence = None

    @staticmethod
    def get_support(name="all", verbose=True):
        if name == "all":
            if verbose:
                for key, value in MhaKCentersClustering.SUPPORT.items():
                    print(f"Supported methods for '{key}' are: ")
                    print(value)
            return MhaKCentersClustering.SUPPORT
        if name in list(MhaKCentersClustering.SUPPORT.keys()):
            if verbose:
                print(f"Supported methods for '{name}' are: ")
                print(MhaKCentersClustering.SUPPORT[name])
            return MhaKCentersClustering.SUPPORT[name]
        raise ValueError(f"MhaKCentersClustering doesn't support {name}.")

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if optimizer_paras is None:
            optimizer_paras = {}
        if type(optimizer) is str:
            opt_class = mu.get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                optimizer = opt_class(**optimizer_paras)
            else:
                optimizer = opt_class(epoch=100, pop_size=20)
        elif isinstance(optimizer, mu.Optimizer):
            if type(optimizer_paras) is dict:
                optimizer.set_parameters(optimizer_paras)
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")
        return optimizer, optimizer_paras

    def _set_obj(self, obj=None):
        if obj in list(self.SUPPORT["obj"].keys()):
            if obj[-1] == "S":
                raise ValueError(f"Invalid obj. MhaKMeansTuner only supports internal objective.")
            return obj
        else:
            print(f"Invalid obj. MhaKMeansTuner doesn't support obj: {obj}")

    def fit(self, data, cluster_finder="elbow", obj="SSEI", verbose=True, mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        data : Data, default=None
            The instance of Data class, make sure you have at least matrix feature X. The target labels y (Optional).
            Also make sure your matrix X is normalized or standardized

        cluster_finder : str, default="elbow".
            The method to find the optimal number of clusters in data. The supported methods are:
            ["elbow", "gap", "silhouette", "davies_bouldin", "calinski_harabasz", "bayesian_ìnormation", "all_min", "all_max", "all_mean", "all_majority"].
            The method has prefixes `all` means that it will try all other methods and get the statistical number of clusters.
            For example, `all_min`, takes the minimum K found from all tried methods. `all_mean`, takes the average K found from all tried methods.

            This parameter is only used when `data.y` is None. If you pass labels `y` to `data`. This method will be turned off.
            The number of clusters will be determined by number of unique labels in `y`.

        obj : str, default=""SSEI""
            A string that represent objective name that supported by the library: https://github.com/thieu1995/permetrics
            To get the supported metrics, please use: MhaKCentersClustering.get_support(), supported obj are supported metrics

        verbose : int, default = True
            Controls verbosity of output for each training process of each optimizer.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) used in Optimizer (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        if data.y is not None:
            n_clusters = len(np.unique(data.y))
        else:
            self.cluster_finder = validator.check_str("cluster_finder", cluster_finder, list(self.SUPPORT["cluster_finder"].keys()))
            n_clusters = getattr(cluster, self.SUPPORT["cluster_finder"][self.cluster_finder])(data.X)
        self.obj = self._set_obj(obj)
        minmax = self.SUPPORT["obj"][self.obj]
        log_to = "console" if verbose else "None"
        lb = np.min(data.X, axis=0).tolist() * n_clusters
        ub = np.max(data.X, axis=0).tolist() * n_clusters
        bound = mu.FloatVar(lb=lb, ub=ub, name="center_weights")
        problem = mu.KCentersClusteringProblem(bounds=bound, minmax=minmax, data=data, obj_name=self.obj, log_to=log_to)
        self.optimizer.solve(problem, mode=mode, n_workers=n_workers, termination=termination, seed=self.seed)
        self.convergence = self.optimizer.history.list_global_best_fit
        self.best_agent = self.optimizer.g_best
        return self

    def predict(self, X):
        if self.best_agent is None:
            raise ValueError(f"The model is not trained yet. You need to call fit() function to train the model.")
        return self.optimizer.problem.get_y_pred(X, self.optimizer.problem.decode_solution(self.best_agent.solution)["center_weights"])
