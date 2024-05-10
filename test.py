from CDC import CDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from precise_check import *
glass=pd.read_csv('glass+identification/glass.csv')

X = glass[ ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

Y = glass[['Type of glass']].values
Y = np.ravel(Y)

time_start = time.time()
res = CDC(30, 0.1, X_pca)
time_end = time.time()
print(time_end-time_start)

print(purity(Y, res))
print(ARI(Y, res))
print(NMI_sklearn(Y, res))
print(NCC(Y, res))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=res, s=10, cmap='hsv', marker='o')
plt.show()

# 使用t-SNE将数据降至3维
# tsne = TSNE(n_components=3, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=res, cmap='viridis')
# plt.title('Clustered Data in 3D t-SNE Space')
# plt.colorbar(scatter)
# plt.show()


# raw_data = pd.read_table('glass+identification\glass.csv', header=None)
# X = np.array(raw_data)
# data = X[:, 1:2]
# ref = X[:, 2]
# time_start = time.time()
# res = CDC(30, 0.1, data)
# time_end = time.time()
# print(time_end-time_start)

# plt.scatter(data[:, 0], data[:, 1], c=res, s=10, cmap='hsv', marker='o')
# plt.show()