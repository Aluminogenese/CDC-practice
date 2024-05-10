import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def ARI(labels_true, labels_pred, beta=1.):
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_true, labels_pred)

    return ari

def NMI_sklearn(labels_true, labels_pred):
    # return metrics.adjusted_mutual_info_score(predict, label)
    return metrics.normalized_mutual_info_score(labels_pred, labels_true)

def NCC(labels_true, labels_pred):
    m = labels_pred.shape[0]
    n = labels_pred.shape[1]
    Y = np.zeros((m, m))
    for r in range(m):
        for s in range(m):
            if labels_true[r] == labels_true[s]:
                Y[r, s] = 1

    drs = np.zeros((m, m))
    for r in range(m):
        for s in range(m):
            for att in range(n):
                if labels_pred[r, att] != labels_pred[s, att]:
                    drs[r, s] += 1

    ncc = 0.0
    for r in range(m):
        for s in range(m):
            if r != s:
                ncc += (n - 2 * drs[r, s]) * Y[r, s] + drs[r, s]

    return ncc
