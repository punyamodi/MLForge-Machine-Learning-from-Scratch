import numpy as np


class NMF:
    def __init__(self, n_components=None, init='random', max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_components if self.n_components is not None else min(n_samples, n_features)
        eps = 1e-10
        W = np.abs(rng.randn(n_samples, k)) + eps
        H = np.abs(rng.randn(k, n_features)) + eps
        prev_err = np.inf
        for _ in range(self.max_iter):
            H = H * (W.T @ X) / (W.T @ W @ H + eps)
            W = W * (X @ H.T) / (W @ H @ H.T + eps)
            err = np.linalg.norm(X - W @ H, 'fro')
            if abs(prev_err - err) < self.tol:
                break
            prev_err = err
        self.components_ = H
        self.W_ = W
        self.reconstruction_err_ = err
        self.n_components_ = k
        return self

    def transform(self, X):
        rng = np.random.RandomState(self.random_state)
        eps = 1e-10
        H = np.abs(rng.randn(self.n_components_, X.shape[1])) + eps
        for _ in range(self.max_iter):
            H = H * (self.W_.T @ X) / (self.W_.T @ self.W_ @ H + eps)
        return H.T

    def fit_transform(self, X):
        self.fit(X)
        return self.W_

    def inverse_transform(self, W):
        return W @ self.components_

    def __repr__(self):
        return f"NMF(n_components={self.n_components}, max_iter={self.max_iter})"
