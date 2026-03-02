import numpy as np


class MeanShift:
    def __init__(self, bandwidth=None, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def _estimate_bandwidth(self, X):
        n = X.shape[0]
        sample_size = min(500, n)
        idx = np.random.choice(n, sample_size, replace=False)
        X_sub = X[idx]
        sq = np.sum(X_sub ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X_sub @ X_sub.T, 0))
        return float(np.std(D) / 2.0) if np.std(D) > 0 else 1.0

    def _gaussian_kernel(self, distances_sq, bandwidth):
        return np.exp(-0.5 * distances_sq / (bandwidth ** 2))

    def fit(self, X):
        if self.bandwidth is None:
            h = self._estimate_bandwidth(X)
        else:
            h = float(self.bandwidth)

        n_samples, n_features = X.shape
        centroids = X.copy().astype(float)

        for _ in range(self.max_iter):
            new_centroids = np.zeros_like(centroids)
            for i in range(n_samples):
                diff = X - centroids[i]
                sq_dist = np.sum(diff ** 2, axis=1)
                weights = self._gaussian_kernel(sq_dist, h)
                weight_sum = weights.sum()
                if weight_sum > 0:
                    new_centroids[i] = np.dot(weights, X) / weight_sum
                else:
                    new_centroids[i] = centroids[i]

            shift = np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
            centroids = new_centroids
            if shift < self.tol:
                break

        unique_centers = []
        for c in centroids:
            if len(unique_centers) == 0:
                unique_centers.append(c)
            else:
                dists = np.sqrt(np.sum((np.array(unique_centers) - c) ** 2, axis=1))
                if np.min(dists) > h / 2.0:
                    unique_centers.append(c)

        self.cluster_centers_ = np.array(unique_centers)

        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            dists = np.sqrt(np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1))
            labels[i] = int(np.argmin(dists))
        self.labels_ = labels
        return self

    def predict(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            dists = np.sqrt(np.sum((self.cluster_centers_ - X[i]) ** 2, axis=1))
            labels[i] = int(np.argmin(dists))
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return (f"MeanShift(bandwidth={self.bandwidth}, "
                f"max_iter={self.max_iter}, tol={self.tol})")
