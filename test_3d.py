import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from CDC import CDC
from sklearn.decomposition import PCA

# 生成三维数据集
X, _ = make_blobs(n_samples=666, centers=6, n_features=9, random_state=42)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用CDC函数进行聚类
k_num = 30  # 聚类数
T_DCM = 0.1  # 阈值
cluster_labels = CDC(k_num, T_DCM, X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=10, cmap='hsv', marker='o')
plt.show()
# 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='viridis')
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
# ax.set_title('Clustered Data in 3D Space')
# plt.show()
