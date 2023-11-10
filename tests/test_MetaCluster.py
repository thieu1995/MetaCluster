#!/usr/bin/env python
# Created by "Thieu" at 17:18, 31/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metacluster import get_dataset, MetaCluster

np.random.seed(42)


def test_MetaCluster_class():
    data = get_dataset("circles")
    data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

    list_optimizer = ["OriginalFBIO", "OriginalGWO", "OriginalSMA"]
    list_paras = [
        {"name": "FBIO", "epoch": 10, "pop_size": 30},
        {"name": "GWO", "epoch": 10, "pop_size": 30},
        {"name": "SMA", "epoch": 10, "pop_size": 30}
    ]
    list_obj = ["BHI", "MIS", "XBI"]

    model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=3, seed=10)
    assert model.n_trials == 3
    assert model.list_obj == list_obj
    assert len(model.list_optimizer) == len(list_optimizer)
