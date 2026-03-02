import numpy as np
from collections import deque


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def _pairwise_distances(self, X):
        if self.metric == 'euclidean':
            sq = np.sum(X ** 2, axis=1)
            D = sq[:, None] + sq[None, :] - 2 * X @ X.T
            return np.sqrt(np.maximum(D, 0))
        if self.metric == 'manhattan':
            n = X.shape[0]
            D = np.zeros((n, n))
            for i in range(n):
                D[i] = np.sum(np.abs(X - X[i]), axis=1)
            return D
        return np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))

    def fit(self, X):
        n = X.shape[0]
        D = self._pairwise_distances(X)
        neighbors = [np.where(D[i] <= self.eps)[0].tolist() for i in range(n)]
        labels = np.full(n, -1, dtype=int)
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbors])
        self.core_sample_indices_ = np.where(is_core)[0]
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            queue = deque([i])
            labels[i] = cluster_id
            while queue:
                pt = queue.popleft()
                for nb in neighbors[pt]:
                    if labels[nb] == -1:
                        labels[nb] = cluster_id
                        if is_core[nb]:
                            queue.append(nb)
            cluster_id += 1
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples})"
