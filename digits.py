import numpy as np
import matplotlib.pyplot as plt
from precise_check import *
from CDC import CDC
import umap
from sklearn.datasets import load_digits
from matplotlib.colors import ListedColormap

digits = load_digits()
data = digits.data
labels = digits.target

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(data)

# 使用CDC函数进行聚类
k_num = 30  # 聚类数
T_DCM = 0.1  # 阈值
res = CDC(k_num, T_DCM, embedding)

print("ARI:")
print(ARI(labels, res))
print("NMI:")
print(NMI_sklearn(labels, res))


# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
# cmap = ListedColormap(colors)
# plt.subplot(1,2,1)
plt.title("prediction")
plt.scatter(embedding[:, 0], embedding[:, 1], c=res, s=10, cmap='Spectral', marker='o')
plt.colorbar(boundaries=np.arange(7) + 0.5).set_ticks(np.arange(1, 7))

# plt.subplot(1,2,2)
# plt.title("true")
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=10, cmap='Spectral', marker='o')
# plt.colorbar(boundaries=np.arange(7) + 0.5).set_ticks(np.arange(1, 7))
plt.savefig('plot.png')
plt.show()