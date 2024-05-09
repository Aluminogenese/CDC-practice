from CDC import CDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

glass=pd.read_csv('glass+identification/glass.csv')

X = glass[ ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']].values
print(X.shape)
Y = glass[['Type of glass']].values
Y = np.ravel(Y)
print(type(X))
print(type(Y))
time_start = time.time()
res = CDC(30, 0.1, X)
time_end = time.time()
print(time_end-time_start)

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