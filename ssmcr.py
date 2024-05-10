from CDC import CDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.manifold import TSNE

glass=pd.read_csv('sepsis+survival+minimal+clinical+records/s41598-020-73558-3_sepsis_survival_validation_cohort.csv')

X = glass[ ['age_years','sex_0male_1female','episode_number']].values
print(X.shape)
Y = glass[['hospital_outcome_1alive_0dead']].values
Y = np.ravel(Y)
print(type(X))
print(type(Y))
time_start = time.time()
res = CDC(30, 0.1, X)
time_end = time.time()
print(time_end-time_start)

# 绘制3D散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=res, cmap='viridis')
plt.title('Clustered Data in 3D t-SNE Space')
plt.colorbar(scatter)
plt.show()