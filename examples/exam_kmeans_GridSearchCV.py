#!/usr/bin/env python
# Created by "Thieu" at 09:38, 10/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.model_selection import GridSearchCV
#
# kmeans = KMeans()
#
# param_grid = {
#     'n_clusters': [2, 3, 4, 5, 6, 7, 8],
#     'init': ['k-means++', 'random'],
#     'max_iter': [100, 200, 300],
#     'n_init': [10, 15, 20],
#     'random_state': [42]
# }
#
# grid_search = GridSearchCV(kmeans, param_grid, cv=5)
# grid_search.fit(X)
#
# best_params = grid_search.best_params_
# print(best_params)
#
# best_kmeans = grid_search.best_estimator_



import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.0)

# Create the KMeans model
kmeans = KMeans()

# Define the parameter grid
param_grid = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
    'init': ['k-means++', 'random'],
    'max_iter': [100, 200, 300],
    'n_init': [10, 15, 20],
    'random_state': [42]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=2, verbose=2)

# Fit the GridSearchCV object on your data
grid_search.fit(X)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(best_params)

# Retrieve the best model
best_kmeans = grid_search.best_estimator_
print(best_kmeans)
