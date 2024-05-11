import numpy as np
from sklearn import metrics

def ARI(labels_true, labels_pred, beta=1.):
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_true, labels_pred)

    return ari

def NMI_sklearn(labels_true, labels_pred):
    # return metrics.adjusted_mutual_info_score(predict, label)
    return metrics.normalized_mutual_info_score(labels_pred, labels_true)
