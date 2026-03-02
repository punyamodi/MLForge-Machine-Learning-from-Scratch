import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward', affinity='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity

    def _pairwise_sq(self, X):
        sq = np.sum(X ** 2, axis=1)
        return sq[:, None] + sq[None, :] - 2 * X @ X.T

    def _pairwise_dist(self, X):
        return np.sqrt(np.maximum(self._pairwise_sq(X), 0))

    def fit(self, X):
        n = X.shape[0]

        cluster_members = {i: [i] for i in range(n)}
        cluster_centers = {i: X[i].copy() for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}
        active = list(range(n))
        next_id = n

        dist = self._pairwise_dist(X)
        np.fill_diagonal(dist, np.inf)

        dist_dict = {}
        for i in active:
            for j in active:
                if i < j:
                    dist_dict[(i, j)] = dist[i, j]

        self.children_ = []

        while len(active) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            for (i, j), d in dist_dict.items():
                if i in active and j in active and d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j

            if merge_i == -1:
                break

            self.children_.append([merge_i, merge_j])

            ni = cluster_sizes[merge_i]
            nj = cluster_sizes[merge_j]
            new_size = ni + nj
            new_center = (ni * cluster_centers[merge_i] + nj * cluster_centers[merge_j]) / new_size
            new_members = cluster_members[merge_i] + cluster_members[merge_j]

            cluster_members[next_id] = new_members
            cluster_centers[next_id] = new_center
            cluster_sizes[next_id] = new_size

            active.remove(merge_i)
            active.remove(merge_j)

            keys_to_remove = [k for k in dist_dict if merge_i in k or merge_j in k]
            for k in keys_to_remove:
                del dist_dict[k]

            for k in active:
                a, b = (k, next_id) if k < next_id else (next_id, k)
                if self.linkage == 'ward':
                    nk = cluster_sizes[k]
                    d_ik = np.sqrt(np.sum((cluster_centers[merge_i] - cluster_centers[k]) ** 2))
                    d_jk = np.sqrt(np.sum((cluster_centers[merge_j] - cluster_centers[k]) ** 2))
                    d_ij = np.sqrt(np.sum((cluster_centers[merge_i] - cluster_centers[merge_j]) ** 2))
                    alpha_i = (ni + nk) / (ni + nj + nk)
                    alpha_j = (nj + nk) / (ni + nj + nk)
                    beta = -nk / (ni + nj + nk)
                    new_d = np.sqrt(max(alpha_i * d_ik**2 + alpha_j * d_jk**2 + beta * d_ij**2, 0))
                elif self.linkage == 'single':
                    members_i = cluster_members[merge_i]
                    members_j = cluster_members[merge_j]
                    members_k = cluster_members[k]
                    all_new = members_i + members_j
                    new_d = np.min([
                        np.sqrt(np.sum((X[a_] - X[b_]) ** 2))
                        for a_ in all_new for b_ in members_k
                    ])
                elif self.linkage == 'complete':
                    members_i = cluster_members[merge_i]
                    members_j = cluster_members[merge_j]
                    members_k = cluster_members[k]
                    all_new = members_i + members_j
                    new_d = np.max([
                        np.sqrt(np.sum((X[a_] - X[b_]) ** 2))
                        for a_ in all_new for b_ in members_k
                    ])
                elif self.linkage == 'average':
                    members_i = cluster_members[merge_i]
                    members_j = cluster_members[merge_j]
                    members_k = cluster_members[k]
                    all_new = members_i + members_j
                    dists_list = [
                        np.sqrt(np.sum((X[a_] - X[b_]) ** 2))
                        for a_ in all_new for b_ in members_k
                    ]
                    new_d = float(np.mean(dists_list))
                else:
                    new_d = np.sqrt(np.sum((new_center - cluster_centers[k]) ** 2))

                dist_dict[(a, b)] = new_d

            active.append(next_id)
            next_id += 1

        self.labels_ = np.zeros(n, dtype=int)
        for label_id, cluster_id in enumerate(active):
            for member in cluster_members[cluster_id]:
                self.labels_[member] = label_id

        self.children_ = np.array(self.children_) if self.children_ else np.empty((0, 2), dtype=int)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return (f"AgglomerativeClustering(n_clusters={self.n_clusters}, "
                f"linkage={self.linkage!r}, affinity={self.affinity!r})")
