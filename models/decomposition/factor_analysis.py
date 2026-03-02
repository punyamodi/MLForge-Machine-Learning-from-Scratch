import numpy as np


class FactorAnalysis:
    def __init__(self, n_components=2, max_iter=1000, tol=1e-2, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_components

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov_X = (X_centered.T @ X_centered) / n_samples

        W = rng.randn(n_features, k) * 0.1
        psi = np.diag(cov_X).copy()
        psi = np.maximum(psi, 1e-6)

        self.loglike_ = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            Psi_diag = psi
            Psi_inv = np.diag(1.0 / Psi_diag)

            M = Psi_inv @ W
            beta_mat = np.linalg.inv(np.eye(k) + W.T @ M)

            Ez = (beta_mat @ M.T @ X_centered.T).T
            Ezzt = n_samples * beta_mat + Ez.T @ Ez

            W_new = X_centered.T @ Ez @ np.linalg.inv(Ezzt)
            psi_new = np.diag(cov_X) - np.sum(W_new * (W.T @ np.diag(cov_X)).T, axis=1)
            psi_new = np.maximum(psi_new, 1e-6)

            Sigma = W_new @ W_new.T + np.diag(psi_new)
            try:
                sign, log_det = np.linalg.slogdet(Sigma)
                if sign <= 0:
                    log_det = np.inf
                Sigma_inv = np.linalg.inv(Sigma)
                ll = -0.5 * n_samples * (n_features * np.log(2 * np.pi)
                                          + log_det
                                          + np.trace(Sigma_inv @ cov_X))
            except np.linalg.LinAlgError:
                ll = -np.inf

            self.loglike_.append(ll)
            W = W_new
            psi = psi_new

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.components_ = W
        self.noise_variance_ = psi
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        psi = self.noise_variance_
        W = self.components_
        k = self.n_components

        Psi_inv = np.diag(1.0 / psi)
        M = Psi_inv @ W
        beta_mat = np.linalg.inv(np.eye(k) + W.T @ M)
        Ez = (beta_mat @ M.T @ X_centered.T).T
        return Ez

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return (f"FactorAnalysis(n_components={self.n_components}, "
                f"max_iter={self.max_iter}, tol={self.tol})")
