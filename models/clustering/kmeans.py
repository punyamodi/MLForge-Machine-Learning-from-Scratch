import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, init='random', n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, X, rng):
        n_samples = X.shape[0]
        if self.init == 'random':
            idx = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[idx].copy()
        centers = [X[rng.randint(0, n_samples)].copy()]
        for _ in range(self.n_clusters - 1):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / dists.sum()
            cumulative = np.cumsum(probs)
            r = rng.rand()
            idx = np.searchsorted(cumulative, r)
            idx = min(idx, n_samples - 1)
            centers.append(X[idx].copy())
        return np.array(centers)

    def _assign(self, X, centers):
        dists = np.sum((X[:, np.newaxis] - centers[np.newaxis]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def _run_once(self, X, rng):
        centers = self._init_centroids(X, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_iter):
            labels = self._assign(X, centers)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centers[k]
                for k in range(self.n_clusters)
            ])
            shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            if shift < self.tol:
                break
        inertia = sum(
            np.sum((X[labels == k] - centers[k]) ** 2)
            for k in range(self.n_clusters)
            if (labels == k).sum() > 0
        )
        return centers, labels, float(inertia)

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        for _ in range(self.n_init):
            centers, labels, inertia = self._run_once(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
        self.n_iter_ = self.max_iter
        return self

    def predict(self, X):
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def transform(self, X):
        return np.sqrt(
            np.sum((X[:, np.newaxis] - self.cluster_centers_[np.newaxis]) ** 2, axis=2)
        )

    def __repr__(self):
        return (f"KMeans(n_clusters={self.n_clusters}, "
                f"init={self.init!r}, n_init={self.n_init})")


class KMeansPlusPlus(KMeans):
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def __repr__(self):
        return f"KMeansPlusPlus(n_clusters={self.n_clusters}, n_init={self.n_init})"
