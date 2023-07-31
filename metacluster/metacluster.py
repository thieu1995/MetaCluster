#!/usr/bin/env python
# Created by "Thieu" at 05:36, 28/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
from pathlib import Path
import pandas as pd
import numpy as np
from metacluster.utils import mealpy_util as mu, cluster
from metacluster.utils import validator
from metacluster.utils.io_util import write_dict_to_csv
from metacluster.utils.mealpy_util import get_all_optimizers, KCenterClusteringProblem
from metacluster.utils.visualize_util import export_boxplot_figures, export_convergence_figures


class MetaCluster:
    """
    Defines a MetaCluster class that hold all Metaheuristic-based K-Center Clustering methods

    Parameters
    ----------

    list_optimizer=None, list_paras=None, list_obj=None, n_trials=5

    list_optimizer: list, default = None
        List of strings that represent class optimizer or list of instance of Optimizer class from Mealpy library.
        Current supported optimizers, please check it here: https://github.com/thieu1995/mealpy
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.
        Please use this to get supported optimizers: MetaCluster.get_support()

    list_paras: list, default=None
        List of dictionaries that present the parameters of each Optimizer class.
        You can set it to None to use all of default parameters in Mealpy library.

    list_obj : list, default=None
        List of strings that represent objective name.
        Current supported objectives, please check it here: https://github.com/thieu1995/permetrics
        Please us this to get supported objectives: MetaCluster.get_support()

    n_trials : int, default=5
        The number of runs for each optimizer for each objective

    Examples
    --------
    The following example shows how to use  the most informative features in the MhaSelector FS method

    >>> import pandas as pd
    >>> from metacluster.wrapper.mha import MhaSelector
    >>> # load dataset
    >>> dataset = pd.read_csv('your_path/dataset.csv', index_col=0).values
    >>> X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column
    >>> # define metacluster feature selection method
    >>> feat_selector = MhaSelector(problem="classification", estimator="rf", optimizer="BaseGA")
    >>> # find all relevant features - 5 features should be selected
    >>> feat_selector.fit(X, y)
    >>> # check selected features - True (or 1) is selected, False (or 0) is not selected
    >>> print(feat_selector.selected_feature_masks)
    array([ True, True, True, False, False, True, False, False, False, True])
    >>> print(feat_selector.selected_feature_solution)
    array([ 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    >>> # check the index of selected features
    >>> print(feat_selector.selected_feature_indexes)
    array([ 0, 1, 2, 5, 9])
    >>> # call transform() on X to filter it down to selected features
    >>> X_filtered = feat_selector.transform(X)
    """

    SUPPORT = {
        "cluster_finder": {"elbow": "get_clusters_by_elbow", "gap": "get_clusters_by_gap_statistic",
                           "silhouette": "get_clusters_by_silhouette_score", "davies_bouldin": "get_clusters_by_davies_bouldin",
                           "calinski_harabasz": "get_clusters_by_calinski_harabasz", "bic": "get_clusters_by_bic",
                           "all_min": "get_clusters_by_all_min", "all_max": "get_clusters_by_all_max",
                           "all_mean": "get_clusters_by_all_mean", "all_majority": "get_clusters_by_all_majority"},
        "obj": {"BHI": "min", "CHI": "max", "XBI": "min", "BRI": "max", "DBI": "min", "DRI": "min",
                "DI": "max", "KDI": "min", "LDRI": "min", "LSRI": "min", "SI": "max", "MIS": "max",
                "NMIS": "max", "RaS": "max", "FMS": "max", "HS": "max", "CS": "max", "VMS": "max",
                "PrS": "max", "ReS": "max", "FmS": "max", "CDS": "max", "HGS": "max", "JS": "max",
                "KS": "max", "MNS": "min", "PhS": "max", "RTS": "max", "RRS": "max", "SS1S": "max", "SS2S": "max"},
        "optimizer": list(get_all_optimizers().keys())
    }

    def __init__(self, list_optimizer=None, list_paras=None, list_obj=None, n_trials=5):
        self.list_optimizer, self.list_paras = self._set_list_optimizer(list_optimizer, list_paras)
        self.list_obj = self._set_list_function(list_obj, name="objectives")
        self.n_trials = n_trials

    @staticmethod
    def get_support():
        for key, value in MetaCluster.SUPPORT.items():
            print(f"Supported values for: {key} are: ")
            print(value)

    def _set_list_function(self, list_obj=None, name="objectives"):
        if type(list_obj) in (list, tuple, np.ndarray):
            list_obj1 = []
            list_obj0 = []
            for obj in list_obj:
                if obj in list(self.SUPPORT["obj"].keys()):
                    list_obj1.append(obj)
                else:
                    list_obj0.append(obj)
            if len(list_obj0) > 0:
                print(f"MetaCluster doesn't support {name}: {list_obj0}")
            return list_obj1

    def _set_list_optimizer(self, list_optimizer=None, list_paras=None):
        if type(list_optimizer) not in (list, tuple):
            raise ValueError("list_optimizers should be a list or tuple.")
        else:
            if list_paras is None or type(list_paras) not in (list, tuple):
                list_paras = [{}, ] * len(list_optimizer)
            elif len(list_paras) != len(list_optimizer):
                raise ValueError("list_paras should be a list with the same length as list_optimizer")
            list_opts = []
            for idx, opt in enumerate(list_optimizer):
                if type(opt) is str:
                    opt_class = mu.get_optimizer_by_name(opt)
                    if type(list_paras[idx]) is dict:
                        list_opts.append(opt_class(**list_paras[idx]))
                    else:
                        list_opts.append(opt_class(epoch=100, pop_size=30))
                elif isinstance(opt, mu.Optimizer):
                    if type(list_paras[idx]) is dict:
                        opt.set_parameters(list_paras[idx])
                    list_opts.append(opt)
                else:
                    raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")
        return list_opts, list_paras

    def __run__(self, model, problem, mode="single", n_workers=2, termination=None):
        _, best_fitness = model.solve(problem, mode=mode, n_workers=n_workers, termination=termination)
        return {
            "best_fitness": best_fitness,
            "best_solution": model.solution,
            "convergence": model.history.list_global_best_fit
        }

    def execute(self, data=None, cluster_finder="elbow", list_metrics=None, save_path="history", save_figures=True,
            verbose=True, mode='single', n_workers=None, termination=None):
        """
        Parameters
        ----------
        data : instance of Data class, default=None
            The instance of Data class, make sure you have at least matrix feature X. Or target labels y (Optional).
            Also make sure your matrix X is normalized or standardized

        cluster_finder : str, default="elbow".
            The method to find the optimal number of clusters in data

        list_metrics : list, default=None
            List of performance metrics that supported by the library: https://github.com/thieu1995/permetrics
            To get the supported metrics, please use: MetaCluster.get_support(), supported obj are supported metrics

        save_path : str, default="history"
            The path to the folder that hold results

        save_figures : bool, default=True
            Save all of available figure or not

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
            cluster_finder = validator.check_str("cluster_finder", cluster_finder, list(self.SUPPORT["cluster_finder"].keys()))
            n_clusters = getattr(cluster, cluster_finder)(data.X)
        lb = np.min(data.X, axis=0).tolist() * n_clusters
        ub = np.max(data.X, axis=0).tolist() * n_clusters
        obj_paras = {"decimal": 4}
        log_to = "console" if verbose else "None"
        list_metrics = self._set_list_function(list_metrics, name="metrics")

        ## Check parent directories
        save_path = f"{save_path}/{data.get_name()}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        list_problems = []
        for idx_opt, opt in enumerate(self.list_optimizer):
            for idx_obj, obj in enumerate(self.list_obj):

                list_dict = []
                for idx_trial, trial in enumerate(range(self.n_trials)):
                    minmax = self.SUPPORT["obj"][obj]
                    prob = KCenterClusteringProblem(lb, ub, minmax, obj_paras=obj_paras, log_to=log_to)
                    list_problems.append(prob)

                    time_run = time.perf_counter()
                    res = self.__run__(opt, prob, mode=mode, n_workers=n_workers, termination=termination)
                    time_run = round(time_run, 5)
                    y_pred = prob.get_y_pred(data.X, res["best_solution"])
                    y_pred = "-".join(y_pred.tolist())      # Convert all labels to single string to save to csv file.
                    conv = "-".join(res["convergence"])
                    dict_metrics = prob.get_metrics(res["best_solution"], list_metrics)

                    ## Save result_labels.csv file
                    dict1 = {"optimizer": opt.get_name(), "obj": obj, "n_clusters": n_clusters, "y_pred": y_pred}
                    write_dict_to_csv(dict1, save_path=save_path, file_name="result_labels.csv")

                    ## Save result_metrics.csv file
                    dict2 = {"optimizer": opt.get_name(), "obj": obj, "trial": trial+1, "n_clusters": n_clusters, "time_run": time_run}
                    dict3 = {**dict2, **dict_metrics}
                    write_dict_to_csv(dict3, save_path=save_path, file_name="result_metrics.csv")

                    ## Save results for metrics-min and metrics-std
                    dict4 = {"time_run": time_run, **dict_metrics}
                    list_dict.append(dict4)

                    ## Save result_convergence.csv
                    dict5 = {"optimizer": opt.get_name(), "obj": obj, "trial": trial+1, "n_clusters": n_clusters, "fitness": conv}
                    write_dict_to_csv(dict5, save_path=save_path, file_name="result_convergences.csv")

                ## Save result_metrics_min.csv and result_metrics_std.csv file
                df0 = pd.DataFrame(list_dict)
                dict_mean = df0.mean().to_dict()
                dict_std = df0.std().to_dict()
                dict_mean = {"optimizer": opt.get_name(), "obj": obj, "n_clusters": n_clusters, **dict_mean}
                dict_std = {"optimizer": opt.get_name(), "obj": obj, "n_clusters": n_clusters, **dict_std}
                write_dict_to_csv(dict_mean, save_path=save_path, file_name="result_metrics_min.csv")
                write_dict_to_csv(dict_std, save_path=save_path, file_name="result_metrics_std.csv")

        if save_figures:
            for idx_metric, metric in enumerate(list_metrics):
                df = pd.read_csv(f"{save_path}/result_metrics.csv", usecols=["optimizer", "obj", metric])
                for idx_obj, obj in enumerate(self.list_obj):
                    df_draw = df[df["obj"] == obj]["optimizer", metric]
                    export_boxplot_figures(df_draw, file_name=f"boxplot-{obj}-{metric}", save_path=save_path)

            df = pd.read_csv(f"{save_path}/result_convergences.csv", usecols=["optimizer", "obj", "trial", "fitness"])
            for idx_obj, obj in enumerate(self.list_obj):
                ## Draw convergence for single trial
                for idx_trial, trial in enumerate(range(self.n_trials)):
                    df_draw = df[(df["obj"] == obj) & (df["trial"] == trial+1)]["optimizer", "fitness"]
                    df_draw.set_index("optimizer", inplace=True)
                    dict_draw = df_draw.to_dict()["fitness"]
                    for key, value in dict_draw.items():
                        dict_draw[key] = np.array(value.split("-"), dtype=float)
                    df_draw = pd.DataFrame(dict_draw)
                    export_convergence_figures(df_draw, file_name=f"convergence-{obj}-{trial+1}", save_path=save_path)

                ## Draw mean convergence of all trials
                df_draw = df[df["obj"] == obj]["optimizer", 'fitness']
                mylist = df_draw.values.tolist()
                dict_mean = {}
                for idx, item in enumerate(mylist):
                    if item[0] in dict_mean:
                        dict_mean[item[0]].append(np.array(item[1].split("-"), dtype=float))
                    else:
                        dict_mean[item[0]] = [np.array(item[1].split("-"), dtype=float)]
                for key, value in dict_mean.items():
                    dict_mean[key] = np.mean(value, axis=0)
                df_draw = pd.DataFrame(dict_mean)
                export_convergence_figures(df_draw, file_name=f"convergence-{obj}-mean", save_path=save_path)
