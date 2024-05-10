import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.special import gamma

def CDC(dat_mat, k, ratio, embedding_method=None, k_UMAP=None, npc=None, norm=False):
    if embedding_method is None:
        embedding_method = "None"
    
    if k_UMAP is None:
        k_UMAP = 30
    
    if norm:
        scaler = MinMaxScaler()
        dat_mat = scaler.fit_transform(dat_mat)
    
    rep_ind = np.where(np.sum(dat_mat[:, :] == dat_mat[:, np.newaxis], axis=1) > 1)[0]
    dat_ind = np.setdiff1d(np.arange(dat_mat.shape[0]), rep_ind)
    
    X = dat_mat[dat_ind, :]
    Y = X
    val_num = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X)
    X = X[:, val_num > 1]
    dim = X.shape[1]
    num = X.shape[0]
    
    if npc is None:
        npc = max(2, min(dim, int(np.log2(k))))
    
    assert k < num
    assert dim >= 2
    assert embedding_method in ["PCA", "UMAP", "None"]
    assert dim >= npc
    
    if embedding_method == "UMAP":
        # Perform UMAP embedding
        import umap
        umap_model = umap.UMAP(n_neighbors=k_UMAP, n_components=npc, random_state=142)
        X = umap_model.fit_transform(X)
        dim = X.shape[1]
        npc = dim
    
    # Search KNN
    knn = NearestNeighbors(n_neighbors=k).fit(X)
    knn_distances, knn_indices = knn.kneighbors(X)
    
    # Calculate PCA DCM
    PCA_DCM = cal_PCA_DCM(X, knn_indices, knn_distances, dim, num, k, npc)
    
    # Calculate reachable distance and generate clusters
    reach_dis = cal_reach_dis(X, knn_indices, knn_distances, PCA_DCM, ratio, num)
    int_clust = con_int_pts(X, reach_dis, num)
    
    # Assign labels to boundary points
    temp_clust = ass_bou_pts(int_clust, reach_dis, num)
    
    # Assign labels to repeated elements
    cluster = ass_rep_pts(dat_mat, Y, rep_ind, dat_ind, temp_clust)
    
    return cluster

def cal_PCA_DCM(X, knn_indices, knn_distances, dim, num, k, npc):
    assert npc <= dim
    PCA_DCM = np.zeros((1, num))
    for i in range(num):
        rel_coor = abs2rel_coor(X, i, knn_indices[i], knn_distances[i], dim, k, npc)
        if dim == 2:
            PCA_DCM[0, i] = cal_2D_DCM(rel_coor, k)
        else:
            facet = ConvexHull(rel_coor).vertices
            PCA_DCM[0, i] = cal_angle_var(facet, rel_coor, npc, k)
    return PCA_DCM

def abs2rel_coor(X, cid, knn_index, knn_dis, dim, k, npc):
    merge_coor = np.vstack((X[cid], X[knn_index]))
    if dim == 2:
        pca_coor = merge_coor
    else:
        pca = PCA(n_components=npc)
        pca_coor = pca.fit_transform(merge_coor)
    c_coor = np.tile(pca_coor[0], (k, 1))
    knn_coor = pca_coor[1:]
    delta_coor = knn_coor - c_coor
    dist_li = np.linalg.norm(delta_coor, axis=1)
    rel_coor = delta_coor / dist_li[:, None]
    return rel_coor

def cal_2D_DCM(rel_coor, k):
    angle = np.arctan2(rel_coor[:, 1], rel_coor[:, 0])
    angle_sort = np.sort(angle)
    angle_diff = np.diff(np.concatenate((angle_sort, [angle_sort[0] + 2 * np.pi])))
    ang_var = np.var(angle_diff)
    return ang_var

def cal_angle_var(facet, rel_coor, dim, k):
    uniq_rel_coor = np.unique(rel_coor, axis=0)
    rep_pts_num = len(rel_coor) - len(uniq_rel_coor)
    angle = []
    for i in range(len(facet)):
        if dim == 2:
            v1 = rel_coor[facet[i][0]]
            v2 = rel_coor[facet[i][1]]
            edg_len = np.linalg.norm(v1 - v2)
            ang_temp = 2 * np.arcsin(edg_len / 2)
        else:
            # Implement the calculation for higher dimensions if needed
            pass
        angle.append(ang_temp)

    total_angle = compute_total_angle(dim)
    if dim > 2:
        angle = np.array(angle) + (total_angle - np.sum(angle)) / len(angle)

    angle = np.array(angle + [0] * (rep_pts_num * (dim - 1)))
    ang_var = np.var(angle) * (len(facet) + rep_pts_num * (dim - 1)) / (total_angle ** 2)

    return ang_var

def cal_reach_dis(X, knn_index, knn_dis, DCM, ratio, num):
    sort_dcm = np.sort(DCM)
    T_DCM = sort_dcm[int(num * ratio)]
    int_bou_mark = np.zeros((num, 1))
    reach_dis = np.zeros((num, 2))
    for i in range(num):
        if DCM[i] < T_DCM:
            int_bou_mark[i] = 1
    int_pts = np.where(int_bou_mark == 1)[0]
    bou_pts = np.where(int_bou_mark == 0)[0]

    for i in range(len(int_pts)):
        curr_knn = knn_index[int_pts[i]]
        nearest_bpts = np.where(int_bou_mark[curr_knn] == 0)[0]
        if len(nearest_bpts) > 0:
            reach_dis[int_pts[i], 0] = np.linalg.norm(X[curr_knn[nearest_bpts[0]]] - X[int_pts[i]])
        else:
            int_coor = np.array([X[int_pts[i]]])
            bou_coor = X[bou_pts]
            int_bou_dis = np.linalg.norm(int_coor - bou_coor, axis=1)
            reach_dis[int_pts[i], 0] = np.min(int_bou_dis)

    for i in range(len(bou_pts)):
        curr_knn = knn_index[bou_pts[i]]
        nearest_bpts = np.where(int_bou_mark[curr_knn] == 1)[0]
        if len(nearest_bpts) > 0:
            reach_dis[bou_pts[i], 0] = np.linalg.norm(X[curr_knn[nearest_bpts[0]]] - X[bou_pts[i]])
        else:
            int_coor = X[int_pts]
            bou_coor = np.array([X[bou_pts[i]]])
            int_bou_dis = np.linalg.norm(int_coor - bou_coor, axis=1)
            reach_dis[bou_pts[i], 0] = int_pts[np.argmin(int_bou_dis)]

    reach_dis[:, 1] = int_bou_mark.flatten()

    return reach_dis

def con_int_pts(X, reach_dis, num):
    int_clust = np.zeros(num)
    clust_id = 1
    int_id = np.where(reach_dis[:, 1] == 1)[0]
    int_dis = np.linalg.norm(X[int_id][:, None] - X[int_id][None, :], axis=2)
    for i in range(len(int_id)):
        ti = int_id[i]
        if int_clust[ti] == 0:
            int_clust[ti] = clust_id
            for j in range(len(int_id)):
                tj = int_id[j]
                if int_dis[i, j] <= reach_dis[ti, 0] + reach_dis[tj, 0]:
                    if int_clust[tj] == 0:
                        int_clust[tj] = clust_id
                    else:
                        temp_clust_id = int_clust[tj]
                        int_clust[np.where(int_clust == temp_clust_id)] = clust_id
            clust_id += 1
    return int_clust

def ass_bou_pts(int_clust, reach_dis, num):
    bou_pts = np.where(reach_dis[:, 1] == 0)[0]
    for pt in bou_pts:
        nearest_int_pt = int(reach_dis[pt, 0])
        int_clust[pt] = int_clust[nearest_int_pt]
    unique_clusters = np.unique(int_clust)
    cluster = np.zeros(num)
    for i, clust_id in enumerate(unique_clusters):
        cluster[np.where(int_clust == clust_id)] = i + 1
    return cluster

def ass_rep_pts(dat_mat, X, rep_ind, dat_ind, temp_clust):
    cluster = np.zeros(len(dat_mat))
    cluster[dat_ind] = temp_clust
    id_map = {tuple(X[i]): temp_clust[i] for i in range(len(X))}
    for i, rep_i in enumerate(rep_ind):
        cluster[rep_i] = id_map[tuple(dat_mat[i])]
    return cluster

def compute_total_angle(dim):
    if dim % 2 == 0:
        S = 2 * np.pi**(dim / 2) / gamma(dim / 2)
    else:
        S = 2**(dim + 1) * np.pi**(dim / 2) / (dim * gamma((dim + 1) / 2))
    return S


# Example usage
data = np.random.rand(100, 10)  # Example data
clusters = CDC(data, k=5, ratio=0.1)
print(clusters)
