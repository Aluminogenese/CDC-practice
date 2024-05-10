from CDC import CDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.manifold import TSNE
from scipy.special import gamma
from sklearn.decomposition import PCA

raw_data = pd.read_table('Iris.txt', header=None)
X = np.array(raw_data)
data = X[:, :4]

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

ref = X[:, 4]

time_start = time.time()
res = CDC(30, 0.1, data_pca)
time_end = time.time()
print(time_end-time_start)

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=res, s=10, cmap='hsv', marker='o')
plt.show()
# 使用t-SNE将数据降至3维
# tsne = TSNE(n_components=3, random_state=42)
# X_tsne = tsne.fit_transform(data)
# # 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=res, cmap='viridis')
# plt.title('Clustered Data in 3D t-SNE Space')
# plt.colorbar(scatter)
# plt.show()