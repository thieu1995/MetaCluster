#!/usr/bin/env python
# Created by "Thieu" at 15:23, 06/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
#
# from metacluster import get_dataset, MetaCluster
# from sklearn.preprocessing import MinMaxScaler
#
# # Get all supported methods and print them out
# MetaCluster.get_support(name="all")
#
# # Scale dataset to range (0, 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = get_dataset("aniso")
# data.X = scaler.fit_transform(data.X)
#
# # Set up Metaheuristic Algorithms
# list_optimizer = ["BaseFBIO", "OriginalGWO", "OriginalSMA"]
# list_paras = [
#     {"name": "FBIO", "epoch": 10, "pop_size": 30},
#     {"name": "GWO", "epoch": 10, "pop_size": 30},
#     {"name": "SMA", "epoch": 10, "pop_size": 30}
# ]
#
# # Set up list objectives and list performance metrics
# list_obj = ["SI", "RSI"]
# list_metric = ["BHI", "DBI", "DI", "CHI", "SSEI", "NMIS", "HS", "CS", "VMS", "HGS"]
#
# # Define MetaCluster model and execute it
# model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=3)
# model.execute(data=data, cluster_finder="elbow", list_metric=list_metric, save_path="history", verbose=False)

__version__ = "1.3.0"

from metacluster.utils.encoder import LabelEncoder
from metacluster.utils.data_loader import Data, get_dataset
from metacluster.metacluster import MetaCluster
from metacluster.mha_cluster import MhaKCentersClustering, MhaKMeansTuner
