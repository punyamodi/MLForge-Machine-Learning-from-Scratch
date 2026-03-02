import numpy as np


class SpectralClustering:
    def __init__(self, n_clusters=8, affinity='rbf', gamma=1.0,
                 n_neighbors=10, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.random_state = random_state

    def _compute_affinity(self, X):
        sq = np.sum(X ** 2, axis=1)
        sq_dist = sq[:, None] + sq[None, :] - 2 * X @ X.T
        sq_dist = np.maximum(sq_dist, 0)
        if self.affinity == 'rbf':
            return np.exp(-self.gamma * sq_dist)
        if self.affinity == 'nearest_neighbors':
            n = X.shape[0]
            dist = np.sqrt(sq_dist)
            A = np.zeros((n, n))
            for i in range(n):
                nn_idx = np.argsort(dist[i])[1: self.n_neighbors + 1]
                A[i, nn_idx] = 1.0
                A[nn_idx, i] = 1.0
            return A
        return np.exp(-self.gamma * sq_dist)

    def _kmeans(self, X, rng):
        n, d = X.shape
        idx = rng.choice(n, self.n_clusters, replace=False)
        centers = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(300):
            dists = np.sum((X[:, None] - centers[None]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[new_labels == k].mean(axis=0) if (new_labels == k).sum() > 0 else centers[k]
                for k in range(self.n_clusters)
            ])
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centers = new_centers
        inertia = sum(
            np.sum((X[labels == k] - centers[k]) ** 2)
            for k in range(self.n_clusters)
            if (labels == k).sum() > 0
        )
        return labels, float(inertia)

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        self.affinity_matrix_ = self._compute_affinity(X)

        A = self.affinity_matrix_
        d = A.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_sym = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

        L_sym = (L_sym + L_sym.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        idx_sorted = np.argsort(eigenvalues)
        embedding = eigenvectors[:, idx_sorted[:self.n_clusters]]

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embedding = embedding / norms

        best_labels = None
        best_inertia = np.inf
        for _ in range(self.n_init):
            labels, inertia = self._kmeans(embedding, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels

        self.labels_ = best_labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return (f"SpectralClustering(n_clusters={self.n_clusters}, "
                f"affinity={self.affinity!r}, gamma={self.gamma})")
