import numpy as np


class TruncatedSVD:
    def __init__(self, n_components=2, n_iter=5, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def _randomized_svd(self, X, k):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        Omega = rng.randn(n_features, k + 10)
        Y = X @ Omega
        for _ in range(self.n_iter):
            Y = X @ (X.T @ Y)
        Q, _ = np.linalg.qr(Y)
        Q = Q[:, :k]
        B = Q.T @ X
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat
        return U[:, :k], S[:k], Vt[:k]

    def fit(self, X):
        U, S, Vt = self._randomized_svd(X, self.n_components)
        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = S ** 2 / (X.shape[0] - 1)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var + 1e-10)
        return self

    def transform(self, X):
        return X @ self.components_.T

    def fit_transform(self, X):
        U, S, Vt = self._randomized_svd(X, self.n_components)
        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = S ** 2 / (X.shape[0] - 1)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var + 1e-10)
        return U * S

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_

    def __repr__(self):
        return f"TruncatedSVD(n_components={self.n_components}, n_iter={self.n_iter})"
