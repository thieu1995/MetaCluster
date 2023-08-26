#!/usr/bin/env python
# Created by "Thieu" at 10:15, 26/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metacluster import get_dataset, MetaCluster
import time

MetaCluster.get_support("cluster_finder")

data = get_dataset("aniso")
data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))
data.y = None

list_optimizer = ["OriginalWOA", "OriginalTLO", ]
list_paras = [
    {"name": "WOA", "epoch": 10, "pop_size": 30},
    {"name": "TLO", "epoch": 10, "pop_size": 30},
]
list_obj = ["BHI"]
list_metric = ["BRI", "DBI", "DRI"]

time_run = time.perf_counter()
model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=2)
model.execute(data=data, cluster_finder="all_majority", list_metric=list_metric, save_path="history", verbose=False)
model.save_boxplots()
model.save_convergences()
print(f"Time process: {time.perf_counter() - time_run} seconds")
