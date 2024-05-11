import numpy as np
from sklearn.datasets import make_blobs

# 生成2维数据
data_2d, labels_2d = make_blobs(n_samples=1000, centers=6, n_features=2, random_state=40,center_box=(0,40))
labels_2d += 1
# 保存数据
np.savetxt('data/2D_data.txt', data_2d, fmt='%.2f', delimiter=',')
np.savetxt('data/2D_labels.txt', labels_2d, fmt='%d')

# 生成n维数据集
data_nd, labels_nd = make_blobs(n_samples=1000, centers=6, n_features=9, random_state=40,center_box=(0,40))
labels_nd += 1
# 保存数据
np.savetxt('data/nD_data.txt', data_nd, fmt='%.2f', delimiter=',')
np.savetxt('data/nD_labels.txt', labels_nd, fmt='%d')