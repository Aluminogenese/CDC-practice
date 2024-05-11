from CDC import CDC
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time
from precise_check import *

# 加载数据
data = np.loadtxt('data/2D_data.txt', delimiter=',')
labels = np.loadtxt('data/2D_labels.txt', dtype=int)

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
data_norm = scaler.fit_transform(data)

time_start = time.time()
res = CDC(30, 0.2, data_norm)
time_end = time.time()
print(time_end-time_start)

print("ARI:")
print(ARI(labels, res))
print("NMI:")
print(NMI_sklearn(labels, res))

plt.title("Classification Result")
plt.scatter(data[:, 0], data[:, 1], c=res, s=10, cmap='Spectral', marker='o')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(7) + 0.5).set_ticks(np.arange(1, 7))

plt.savefig('results/2d.png')
plt.show()