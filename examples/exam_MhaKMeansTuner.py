#!/usr/bin/env python
# Created by "Thieu" at 11:15, 10/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from metacluster import get_dataset, MhaKMeansTuner
import time

data = get_dataset("aggregation")
data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

# Get all supported methods and print them out
MhaKMeansTuner.get_support(name="all")

time_run = time.perf_counter()
model = MhaKMeansTuner(optimizer="OriginalWOA", optimizer_paras={"name": "GWO", "epoch": 10, "pop_size": 20}, seed=10)
model.fit(data.X, mealpy_bound=None, max_clusters=5, obj="SSEI", verbose=True)

print(model.best_parameters)
print(model.best_estimator.predict(data.X))
print(model.predict(data.X))

print(f"Time process: {time.perf_counter() - time_run} seconds")
