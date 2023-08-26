#!/usr/bin/env python
# Created by "Thieu" at 17:39, 30/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from permetrics import ClusteringMetric

DEFAULT_LIST_CLUSTERS = list(range(2, 11))


def get_all_clustering_metrics():
    dict_results = {}
    for key, value in ClusteringMetric.SUPPORT.items():
        if value["type"] in ("min", "max"):
            dict_results[key] = value["type"]
    return dict_results


### ELBOW
def get_clusters_by_elbow(X, list_clusters=None, **kwargs):
    """
    1. First, apply K-means clustering to the dataset for a range of different values of K, where K is the number of clusters. For example, you might try K=1,2,3,...,10.
    2. For each value of K, compute the Sum of Squared Errors (SSE), which is the sum of the squared distances between each data point and its assigned centroid. The SSE can be obtained from the KMeans object's `inertia_` attribute.
    3. Plot the SSE for each value of K. You should see that the SSE decreases as K increases, because as K increases, the centroids are closer to the data points. However, at some point, increasing K further will not improve the SSE as much. The idea of the elbow method is to identify the value of K at which the SSE starts to level off or decrease less rapidly, forming an "elbow" in the plot. This value of K is considered the optimal number of clusters.
    """
    if type(list_clusters) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_clusters]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS
    wcss = []
    for n_c in list_clusters:
        kmeans = KMeans(n_clusters=n_c)
        kmeans.fit(X=X)
        wcss.append(kmeans.inertia_)
    x1, y1 = 2, wcss[0]
    x2, y2 = list_clusters[-1], wcss[-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)
    return distances.index(max(distances)) + 2
### END ELBOW


def compute_gap_statistic(X, refs=None, B=10, list_K=None, N_init=10):
    """
    This function first generates B reference samples; for each sample, the sample size is the same as the original datasets;
    the value for each reference sample follows a uniform distribution for the range of each feature of the original datasets;
    using simplify formula to compute the D of each cluster, and then the Wk; K should be a increment list, 1-10 is fair enough;
    the B value is about the number of replicated samples to run gap-statistics,
    it is recommended as 10, and it should not be changed/decreased that to a smaller value;

    Parameters
    ----------
    X :  np.array, the original data;
    refs : np.ndarray or None, it is the replicated data that you want to compare with if there exists one; if no existing replicated/proper data, just use None, and the function  will automatically generates them;
    B : int, the number of replicated samples to run gap-statistics; it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    K : list type, the range of K values to test on;
    N_init : int, states the number of initial starting points for each K-mean running under sklearn, in order to get stable clustering result each time;

    Returns
    -------
        gaps: np.array, containing all the gap-statistics results;
        s_k: float, the baseline value to minus with; say reference paper for detailed meaning;
        K: list, containing all the tested K values;
    """
    shape = X.shape
    if refs == None:
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = np.matrix(np.diag(tops - bots))
        rands = np.random.random_sample(size=(shape[0], shape[1], B))
        for i in range(B):
            rands[:, :, i] = rands[:, :, i] * dists + bots
    else:
        rands = refs

    if type(list_K) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_K]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS

    gaps = np.zeros(len(list_clusters))
    Wks = np.zeros(len(list_clusters))
    Wkbs = np.zeros((len(list_clusters), B))

    for indk, k in enumerate(list_clusters):
        # n_init is the number of times each Kmeans running to get stable clustering results under each K value
        k_means = KMeans(n_clusters=k, init='k-means++', n_init=N_init, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
        k_means.fit(X)
        classification_result = k_means.labels_
        # compute the Wk for the classification result
        Wks[indk] = compute_Wk(X, classification_result)

        # clustering on B reference datasets for each 'k'
        for i in range(B):
            Xb = rands[:, :, i]
            k_means.fit(Xb)
            classification_result_b = k_means.labels_
            Wkbs[indk, i] = compute_Wk(Xb, classification_result_b)

    # compute gaps and sk
    gaps = (np.log(Wkbs)).mean(axis=1) - np.log(Wks)
    sd_ks = np.std(np.log(Wkbs), axis=1)
    sk = sd_ks * np.sqrt(1 + 1.0 / B)
    return gaps, sk, list_clusters


### gap statistics
def get_clusters_by_gap_statistic(X, list_clusters=None, B=10, N_init=10, **kwargs):
    gaps, s_k, K = compute_gap_statistic(X, refs=None, B=B, list_K=list_clusters, N_init=N_init)
    gaps_thres = gaps - s_k
    below_or_above = (gaps[0:-1] >= gaps_thres[1:])
    if below_or_above.any():
        optimal_k = K[below_or_above.argmax()]
    else:
        optimal_k = K[-1]
    return optimal_k


def compute_Wk(data: np.ndarray, classification_result: np.ndarray):
    """
    This function computes the Wk after each clustering

    Parameters
    ----------
    data : np.array, containing all the data
    classification_result : np.ndarray, containing all the clustering results for all the data

    Returns
    -------
    Wk : float
    """
    Wk = 0
    label_set = set(classification_result)
    for label in label_set:
        each_cluster = data[classification_result == label, :]
        mu = each_cluster.mean(axis=0)
        D = sum(sum((each_cluster - mu) ** 2)) * 2.0 * each_cluster.shape[0]
        Wk = Wk + D / (2.0 * each_cluster.shape[0])
    return Wk


### silhouette score
def get_clusters_by_silhouette_score(X, list_clusters=None, **kwargs):
    if type(list_clusters) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_clusters]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS
    sil_max = 0
    sil_max_clusters = 2
    for n_clusters in list_clusters:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        sil_score = metrics.silhouette_score(X, labels)
        if sil_score > sil_max:
            sil_max = sil_score
            sil_max_clusters = n_clusters
    return sil_max_clusters
### END silhouette score


### DB score
def get_clusters_by_davies_bouldin(X, list_clusters=None, **kwargs):
    if type(list_clusters) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_clusters]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS
    list_dbs = []
    for n_clusters in list_clusters:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        db_score = metrics.davies_bouldin_score(X, labels)
        list_dbs.append(db_score)
    return list_clusters[np.argmin(list_dbs)]
### END DB score


### Calinski-Harabasz Index
def get_clusters_by_calinski_harabasz(X, list_clusters=None, **kwargs):
    if type(list_clusters) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_clusters]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS
    list_chs = []
    for n_clusters in list_clusters:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        ch_score = metrics.calinski_harabasz_score(X, labels)
        list_chs.append(ch_score)
    return list_clusters[np.argmax(list_chs)]
### END Calinski-Harabasz Index


### Bayesian Information Criterion
def get_clusters_by_bic(X, list_clusters=None, **kwargs):
    if type(list_clusters) in (list, tuple, np.ndarray):
        list_clusters = [item for item in list_clusters]
    else:
        list_clusters = DEFAULT_LIST_CLUSTERS
    bic_max = 0
    bic_max_clusters = 2
    for n_clusters in list_clusters:
        gm = GaussianMixture(n_components=n_clusters, n_init=10, tol=1e-3, max_iter=1000).fit(X)
        bic_score = -gm.bic(X)
        if bic_score > bic_max:
            bic_max = bic_score
            bic_max_clusters = n_clusters
    return bic_max_clusters
### END Bayesian Information Criterion


def compute_all_methods(X, list_clusters=None, **kwargs):
    k1 = get_clusters_by_elbow(X, list_clusters, **kwargs)
    k2 = get_clusters_by_gap_statistic(X, list_clusters, **kwargs)
    k3 = get_clusters_by_silhouette_score(X, list_clusters, **kwargs)
    k4 = get_clusters_by_davies_bouldin(X, list_clusters, **kwargs)
    k5 = get_clusters_by_calinski_harabasz(X, list_clusters, **kwargs)
    k6 = get_clusters_by_bic(X, list_clusters, **kwargs)
    return [k1, k2, k3, k4, k5, k6]


def get_clusters_all_min(X, list_clusters=None, **kwargs):
    k_list = compute_all_methods(X, list_clusters, **kwargs)
    return min(k_list)


def get_clusters_all_max(X, list_clusters=None, **kwargs):
    k_list = compute_all_methods(X, list_clusters, **kwargs)
    return max(k_list)


def get_clusters_all_mean(X, list_clusters=None, **kwargs):
    k_list = compute_all_methods(X, list_clusters, **kwargs)
    return int(np.mean(k_list))


def get_clusters_all_majority(X, list_clusters=None, **kwargs):
    k_list = compute_all_methods(X, list_clusters, **kwargs)
    return max(set(k_list), key=k_list.count)
