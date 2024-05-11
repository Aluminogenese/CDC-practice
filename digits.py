import numpy as np
import matplotlib.pyplot as plt
from precise_check import *
from CDC import CDC
import umap
from sklearn.datasets import load_digits
import time

digits = load_digits()
data = digits.data
labels = digits.target
labels += 1

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(data)

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
plt.colorbar(boundaries=np.arange(19)+0.5).set_ticks(np.arange(1,19))

plt.savefig('results/digits.png')
plt.show()