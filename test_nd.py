import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from CDC import CDC
import umap
import time
from precise_check import *

# 加载数据
data = np.loadtxt('data/nD_data.txt', delimiter=',')
labels = np.loadtxt('data/nD_labels.txt', dtype=int)

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
data_norm = scaler.fit_transform(data)

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(data_norm)

# 使用CDC函数进行聚类
time_start = time.time()
k_num = 30  # 聚类数
T_DCM = 0.3  # 阈值
res = CDC(k_num, T_DCM, embedding)
time_end = time.time()
print(time_end-time_start)

print("ARI:")
print(ARI(labels, res))
print("NMI:")
print(NMI_sklearn(labels, res))

plt.title("Classification Result")
plt.scatter(embedding[:, 0], embedding[:, 1], c=res, s=10, cmap='Spectral', marker='o')
plt.colorbar(boundaries=np.arange(7) + 0.5).set_ticks(np.arange(1, 7))
plt.savefig('results/nd.png')
plt.show()