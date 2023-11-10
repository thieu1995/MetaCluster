#!/usr/bin/env python
# Created by "Thieu" at 20:33, 10/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metacluster import get_dataset, MhaKCentersClustering
import time

data = get_dataset("aggregation")
data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

# Get all supported methods and print them out
MhaKCentersClustering.get_support(name="all")

time_run = time.perf_counter()
model = MhaKCentersClustering(optimizer="OriginalWOA", optimizer_paras={"name": "GWO", "epoch": 10, "pop_size": 20}, seed=10)
model.fit(data, cluster_finder="elbow", obj="SSEI", verbose=True)

print(model.best_agent)
print(model.convergence)
print(model.predict(data.X))

print(f"Time process: {time.perf_counter() - time_run} seconds")
