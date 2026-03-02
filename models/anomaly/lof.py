import numpy as np


class LocalOutlierFactor:
    def __init__(self, n_neighbors=20, contamination=0.1, metric='euclidean', novelty=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty

    def _pairwise_distances(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        cross = X1 @ X2.T
        dists = X1_sq + X2_sq.T - 2 * cross
        dists = np.maximum(dists, 0.0)
        return np.sqrt(dists)

    def fit(self, X):
        self.X_train_ = X.copy()
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        self._k = k

        D = self._pairwise_distances(X, X)

        self.neighbors_ = np.zeros((n, k), dtype=int)
        self.k_distances_ = np.zeros(n)

        for i in range(n):
            sorted_idx = np.argsort(D[i])
            sorted_idx = sorted_idx[sorted_idx != i]
            sorted_idx = sorted_idx[:k]
            self.neighbors_[i] = sorted_idx
            self.k_distances_[i] = D[i, sorted_idx[-1]]

        reach_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                reach_dist[i, j] = max(self.k_distances_[j], D[i, j])

        self.lrd_ = np.zeros(n)
        for i in range(n):
            sum_rd = np.sum(reach_dist[i, self.neighbors_[i]])
            if sum_rd == 0:
                self.lrd_[i] = np.inf
            else:
                self.lrd_[i] = k / sum_rd

        self.lof_ = np.zeros(n)
        for i in range(n):
            neighbor_lrds = self.lrd_[self.neighbors_[i]]
            if self.lrd_[i] == 0:
                self.lof_[i] = np.inf
            else:
                self.lof_[i] = np.mean(neighbor_lrds) / self.lrd_[i]

        self.negative_outlier_factor_ = -self.lof_
        self.threshold_ = np.percentile(self.lof_, 100 * (1 - self.contamination))

        if not self.novelty:
            self.offset_ = -np.percentile(-self.negative_outlier_factor_, 100 * self.contamination)

        return self

    def _compute_lof_scores(self, X):
        n_train = self.X_train_.shape[0]
        n_test = X.shape[0]
        k = self._k

        D = self._pairwise_distances(X, self.X_train_)

        neighbors = np.zeros((n_test, k), dtype=int)
        k_distances = np.zeros(n_test)

        for i in range(n_test):
            sorted_idx = np.argsort(D[i])[:k]
            neighbors[i] = sorted_idx
            k_distances[i] = D[i, sorted_idx[-1]]

        lrd_test = np.zeros(n_test)
        for i in range(n_test):
            sum_rd = 0.0
            for j in neighbors[i]:
                sum_rd += max(self.k_distances_[j], D[i, j])
            if sum_rd == 0:
                lrd_test[i] = np.inf
            else:
                lrd_test[i] = k / sum_rd

        lof_test = np.zeros(n_test)
        for i in range(n_test):
            neighbor_lrds = self.lrd_[neighbors[i]]
            if lrd_test[i] == 0:
                lof_test[i] = np.inf
            else:
                lof_test[i] = np.mean(neighbor_lrds) / lrd_test[i]

        return lof_test

    def decision_function(self, X):
        if not self.novelty:
            return self.negative_outlier_factor_ + self.offset_
        lof_scores = self._compute_lof_scores(X)
        return -(lof_scores - self.threshold_)

    def fit_predict(self, X):
        self.fit(X)
        return np.where(self.lof_ > self.threshold_, -1, 1)

    def predict(self, X):
        if not self.novelty:
            raise ValueError("predict is only available in novelty=True mode. Use fit_predict for training data.")
        lof_scores = self._compute_lof_scores(X)
        return np.where(lof_scores > self.threshold_, -1, 1)

    def score_samples(self, X):
        if not self.novelty:
            return self.negative_outlier_factor_
        lof_scores = self._compute_lof_scores(X)
        return -lof_scores

    def __repr__(self):
        return f"LocalOutlierFactor(n_neighbors={self.n_neighbors}, contamination={self.contamination}, novelty={self.novelty})"
