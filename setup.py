#!/usr/bin/env python
# Created by "Thieu" at 13:24, 25/05/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import setuptools
import os
import re


with open("requirements.txt", encoding='utf-8') as f:
    REQUIREMENTS = f.read().splitlines()


def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'metacluster', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        init_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", init_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('README.md', encoding='utf-8') as f:
        res = f.read()
    return res

setuptools.setup(
    name="metacluster",
    version=get_version(),
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="MetaCluster: An Open-Source Python Library for Metaheuristic-based Clustering Problems",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["clustering", "optimization", "k-center clustering", "data points", "centers", "euclidean distance", "maximum distance",
              "NP-hard", "greedy algorithm", "approximation algorithm", "covering problem", "computational complexity",
              "geometric algorithms", "machine learning", "pattern recognition", "spatial analysis", "graph theory", "mathematical optimization",
              "dimensionality reduction", "mutual information", "correlation-based feature selection",
              "Genetic algorithm (GA)", "Particle swarm optimization (PSO)", "Ant colony optimization (ACO)",
              "Differential evolution (DE)", "Simulated annealing", "Grey wolf optimizer (GWO)", "Whale Optimization Algorithm (WOA)",
              "confusion matrix", "recall", "precision", "accuracy", "K-Nearest Neighbors",
              "pearson correlation coefficient (PCC)", "spearman correlation coefficient (SCC)",
              "multi-objectives optimization problems", "Stochastic optimization", "Global optimization",
              "Convergence analysis", "Search space exploration", "Local search", "Computational intelligence", "Robust optimization",
              "Performance analysis", "Intelligent optimization", "Simulations"],
    url="https://github.com/thieu1995/metacluster",
    project_urls={
        'Documentation': 'https://metacluster.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/metacluster',
        'Bug Tracker': 'https://github.com/thieu1995/metacluster/issues',
        'Change Log': 'https://github.com/thieu1995/metacluster/blob/master/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=setuptools.find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": ["pytest>=7.1.2", "twine>=4.0.1", "pytest-cov==4.0.0", "flake8>=4.0.1"],
    },
    python_requires='>=3.7',
)
