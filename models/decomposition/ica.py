import numpy as np


class IndependentComponentAnalysis:
    def __init__(self, n_components=None, fun='logcosh', max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _g_and_gprime(self, u):
        if self.fun == 'logcosh':
            g = np.tanh(u)
            g_prime = 1 - np.tanh(u) ** 2
        elif self.fun == 'exp':
            exp_u = np.exp(-u ** 2 / 2)
            g = u * exp_u
            g_prime = (1 - u ** 2) * exp_u
        else:
            raise ValueError(f"Unknown function: {self.fun}")
        return g, g_prime

    def _sym_decorrelation(self, W):
        vals, vecs = np.linalg.eigh(W @ W.T)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10)))
        return vecs @ D_inv_sqrt @ vecs.T @ W

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_comp = self.n_components if self.n_components is not None else n_features

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = X_centered.T @ X_centered / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        pos_mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_mask]
        eigenvectors = eigenvectors[:, pos_mask]
        n_comp = min(n_comp, len(eigenvalues))

        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues[:n_comp]))
        self.whitening_ = D_inv_sqrt @ eigenvectors[:, :n_comp].T
        X_white = (self.whitening_ @ X_centered.T).T

        W = rng.randn(n_comp, n_comp)
        W = self._sym_decorrelation(W)

        for _ in range(self.max_iter):
            WX = W @ X_white.T
            g, g_prime = self._g_and_gprime(WX)
            W_new = (g @ X_white) / n_samples - g_prime.mean(axis=1, keepdims=True) * W
            W_new = self._sym_decorrelation(W_new)
            convergence = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if convergence < self.tol:
                break

        self.components_ = W
        self.mixing_ = np.linalg.pinv(W)
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_white = (self.whitening_ @ X_centered.T).T
        return (self.components_ @ X_white.T).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return (f"IndependentComponentAnalysis(n_components={self.n_components}, "
                f"fun='{self.fun}', max_iter={self.max_iter})")
