
<p align="center">
<img style="max-width:100%;" 
src="https://thieu1995.github.io/post/2023-08/MetaCluster-01.png" 
alt="MetaCluster"/>
</p>

---

[![GitHub release](https://img.shields.io/badge/release-1.1.0-yellow.svg)](https://github.com/thieu1995/metacluster/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/metacluster) 
[![PyPI version](https://badge.fury.io/py/metacluster.svg)](https://badge.fury.io/py/metacluster)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metacluster.svg)
![PyPI - Status](https://img.shields.io/pypi/status/metacluster.svg)
[![Downloads](https://static.pepy.tech/badge/MetaCluster)](https://pepy.tech/project/MetaCluster)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/metacluster/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/metacluster/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/metacluster.svg)
[![Documentation Status](https://readthedocs.org/projects/metacluster/badge/?version=latest)](https://metacluster.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/metacluster.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/670197315.svg)](https://zenodo.org/badge/latestdoi/670197315)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MetaCluster is the largest open-source nature-inspired optimization (Metaheuristic Algorithms) library for 
clustering problem in Python

* **Free software:** GNU General Public License (GPL) V3 license
* **Total nature-inspired metaheuristic optimizers (Metaheuristic Algorithms)**: > 200 optimizers
* **Total objective functions (as fitness)**: > 40 objectives
* **Total supported datasets**: 48 datasets from Scikit learn, UCI, ELKI, KEEL...
* **Total performance metrics**: > 40 metrics
* **Total different way of detecting the K value**: >= 10 methods
* **Documentation:** https://metacluster.readthedocs.io/en/latest/
* **Python versions:** >= 3.7.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, plotly, kaleido


# Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/metacluster):
```sh 
$ pip install metacluster==1.1.0
```

* Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/metacluster.git
$ cd metacluster
$ python setup.py install
```

* In case, you want to install the development version from Github:
```sh 
$ pip install git+https://github.com/thieu1995/permetrics 
```

After installation, you can import MetaCluster as any other Python module:

```sh
$ python
>>> import metacluster
>>> metacluster.__version__
```

### Examples

Let's go through some examples.

#### 1. First, load dataset. You can use the available datasets from MetaCluster:

```python 
# Load available dataset from MetaCluster
from metacluster import get_dataset

# Try unknown data
get_dataset("unknown")
# Enter: 1      -> This wil list all of avaialble dataset

data = get_dataset("Arrhythmia")
```

* Or you can load your own dataset 

```python
import pandas as pd
from metacluster import Data

# load X and y
# NOTE MetaCluster accepts numpy arrays only, hence use the .values attribute
dataset = pd.read_csv('examples/dataset.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y, name="my-dataset")
```

#### 2. Next, scale your features

**You should confirm that your dataset is scaled and normalized**

```python 
# MinMaxScaler 
data.X, scaler = data.scale(data.X, method="MinMaxScaler", feature_range=(0, 1))

# StandardScaler 
data.X, scaler = data.scale(data.X, method="StandardScaler")

# MaxAbsScaler 
data.X, scaler = data.scale(data.X, method="MaxAbsScaler")

# RobustScaler 
data.X, scaler = data.scale(data.X, method="RobustScaler")

# Normalizer 
data.X, scaler = data.scale(data.X, method="Normalizer", norm="l2")   # "l1" or "l2" or "max"
```


#### 3. Next, select Metaheuristic Algorithm, Its parameters, list of objectives, and list of performance metrics 

```python 
list_optimizer = ["BaseFBIO", "OriginalGWO", "OriginalSMA"]
list_paras = [
    {"name": "FBIO", "epoch": 10, "pop_size": 30},
    {"name": "GWO", "epoch": 10, "pop_size": 30},
    {"name": "SMA", "epoch": 10, "pop_size": 30}
]
list_obj = ["SI", "RSI"]
list_metric = ["BHI", "DBI", "DI", "CHI", "SSEI", "NMIS", "HS", "CS", "VMS", "HGS"]
```

You can check all supported metaheuristic algorithms from: https://github.com/thieu1995/mealpy.
All supported clustering objectives and metrics from: https://github.com/thieu1995/permetrics.

If you don't want to read the documents, you can print out all of the supported information by:

```python 
from metacluster import MetaCluster 

# Get all supported methods and print them out
MetaCluster.get_support(name="all")
```


#### 4. Next, create an instance of MetaCluster class and run it.

```python 
model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=3)

model.execute(data=data, cluster_finder="elbow", list_metric=list_metric, save_path="history", verbose=False)

model.save_boxplots()
model.save_convergences()

```

As you can see, you can define different datasets and using the same model to run it. 
Remember to set the name to your dataset, because the folder that hold your results is the name of your dataset.


# Support 

### Official links (questions, problems)

* Official source code repo: https://github.com/thieu1995/metacluster
* Official document: https://metacluster.readthedocs.io/
* Download releases: https://pypi.org/project/metacluster/
* Issue tracker: https://github.com/thieu1995/metacluster/issues
* Notable changes log: https://github.com/thieu1995/metacluster/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are optimization and machine learning. Check it here:
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/mafese
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/aiir-team


### Citation Request

Please include these citations if you plan to use this library:

```code 
@software{van_thieu_nguyen_2023_8220709,
  author       = {Nguyen Van Thieu},
  title        = {MetaCluster: An Open-Source Python Library for Metaheuristic-based Clustering Problems},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8214539},
  url          = {https://github.com/thieu1995/metacluster}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}
```

### Supported links 

```code 
1. https://jtemporal.com/kmeans-and-elbow-method/
2. https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
3. https://github.com/minddrummer/gap/blob/master/gap/gap.py
4. https://www.tandfonline.com/doi/pdf/10.1080/03610927408827101
5. https://doi.org/10.1016/j.engappai.2018.03.013
6. https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb
7. https://elki-project.github.io/
8. https://sci2s.ugr.es/keel/index.php
9. https://archive.ics.uci.edu/datasets
10. https://python-charts.com/distribution/box-plot-plotly/
11. https://plotly.com/python/box-plots/?_ga=2.50659434.2126348639.1688086416-114197406.1688086416#box-plot-styling-mean--standard-deviation
```
