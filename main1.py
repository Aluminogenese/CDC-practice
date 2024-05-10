import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.datasets import load_iris
from umap import UMAP
from CDC1 import CDC

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# UMAP Embedding
n_neighbors = 25
n_components = 2
umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=142)
umap_dat = umap_model.fit_transform(X_normalized)

# Run CDC algorithm
k = 8
ratio = 0.75
res = CDC(umap_dat, k, ratio)
clus = res.astype(int)

# Calculate the validity metrics
ari = adjusted_rand_score(y, clus)
acc = accuracy_score(y, clus)

