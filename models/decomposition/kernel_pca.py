import numpy as np


class KernelPCA:
    def __init__(self, n_components=None, kernel='rbf', gamma=1.0, degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _compute_kernel(self, X, Y):
        if self.kernel == 'rbf':
            sq_dists = (np.sum(X ** 2, axis=1, keepdims=True)
                        + np.sum(Y ** 2, axis=1)
                        - 2 * X @ Y.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.gamma * sq_dists)
        elif self.kernel == 'poly':
            return (self.gamma * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'cosine':
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
            return X_norm @ Y_norm.T
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X):
        self.X_fit_ = X
        n = X.shape[0]
        K = self._compute_kernel(X, X)
        one_n = np.ones((n, n)) / n
        self.K_col_mean_ = K.mean(axis=0)
        self.K_grand_mean_ = K.mean()
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        pos_mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_mask]
        eigenvectors = eigenvectors[:, pos_mask]

        n_comp = self.n_components if self.n_components is not None else len(eigenvalues)
        n_comp = min(n_comp, len(eigenvalues))

        self.lambdas_ = eigenvalues[:n_comp]
        self.alphas_ = eigenvectors[:, :n_comp] / np.sqrt(self.lambdas_)
        self.n_components_ = n_comp
        return self

    def transform(self, X):
        n_fit = self.X_fit_.shape[0]
        K_test = self._compute_kernel(X, self.X_fit_)
        K_test_centered = (K_test
                           - K_test.mean(axis=1, keepdims=True)
                           - self.K_col_mean_
                           + self.K_grand_mean_)
        return K_test_centered @ self.alphas_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return (f"KernelPCA(n_components={self.n_components}, kernel='{self.kernel}', "
                f"gamma={self.gamma})")
