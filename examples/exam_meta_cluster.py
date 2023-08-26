#!/usr/bin/env python
# Created by "Thieu" at 17:03, 24/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metacluster import get_dataset, MetaCluster
import time

data = get_dataset("circles")
data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

# Get all supported methods and print them out
MetaCluster.get_support(name="all")

list_optimizer = ["BaseFBIO", "OriginalGWO", "OriginalSMA", "OriginalWOA", "OriginalTLO", "OriginalALO", "OriginalAOA", "OriginalBBO",
                  "OriginalBMO", "BaseGCO", "OriginalHGSO", "OriginalHHO", "OriginalICA"]
list_paras = [
    {"name": "FBIO", "epoch": 10, "pop_size": 30},
    {"name": "GWO", "epoch": 10, "pop_size": 30},
    {"name": "SMA", "epoch": 10, "pop_size": 30},
    {"name": "WOA", "epoch": 10, "pop_size": 30},
    {"name": "TLO", "epoch": 10, "pop_size": 30},
    {"name": "ALO", "epoch": 10, "pop_size": 30},
    {"name": "AOA", "epoch": 10, "pop_size": 30},
    {"name": "BBO", "epoch": 10, "pop_size": 30},
    {"name": "BMO", "epoch": 10, "pop_size": 30},
    {"name": "GCO", "epoch": 10, "pop_size": 30},
    {"name": "HGSO", "epoch": 10, "pop_size": 30},
    {"name": "HHO", "epoch": 10, "pop_size": 30},
    {"name": "ICA", "epoch": 10, "pop_size": 30},
]
list_obj = ["BHI", "MIS", "XBI"]
list_metric = ["BRI", "DBI", "DRI", "DI", "KDI"]

time_run = time.perf_counter()
model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=2)
model.execute(data=data, cluster_finder="elbow", list_metric=list_metric, save_path="history", verbose=False)
model.save_boxplots(figure_size=None, xlabel="Optimizer", list_ylabel=None, title="Boxplot of comparison models",
                      show_legend=True, show_mean_only=False, exts=(".png", ".pdf"), file_name="boxplot")
model.save_convergences(figure_size=None, xlabel="Epoch", list_ylabel=None,
                          title="Convergence chart of comparison models", exts=(".png", ".pdf"), file_name="convergence")
print(f"Time process: {time.perf_counter() - time_run} seconds")
