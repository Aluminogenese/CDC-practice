import numpy as np
from sklearn import metrics

def ARI(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def NMI_sklearn(labels_true, labels_pred):
    return metrics.normalized_mutual_info_score(labels_pred, labels_true)
