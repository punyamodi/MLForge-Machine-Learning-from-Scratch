import numpy as np


class OneClassSVM:
    def __init__(self, kernel='rbf', nu=0.5, gamma='scale', degree=3, coef0=0.0, tol=1e-3, max_iter=1000, random_state=None):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_gamma(self, X):
        if self.gamma == 'scale':
            var = X.var()
            if var == 0:
                return 1.0
            return 1.0 / (X.shape[1] * var)
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        else:
            return float(self.gamma)

    def _kernel_matrix(self, X1, X2, gamma=None):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
            cross = X1 @ X2.T
            sq_dists = X1_sq + X2_sq.T - 2 * cross
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'poly':
            return (gamma * X1 @ X2.T + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(gamma * X1 @ X2.T + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state if self.random_state is not None else 42)

        gamma_val = self._get_gamma(X)
        self.gamma_ = gamma_val

        K = self._kernel_matrix(X, X, gamma_val)

        upper = 1.0 / (self.nu * n)
        alpha = np.full(n, self.nu / n)

        for iteration in range(self.max_iter):
            i, j = rng.choice(n, size=2, replace=False)

            g_i = K[i] @ alpha
            g_j = K[j] @ alpha

            kii = K[i, i]
            kjj = K[j, j]
            kij = K[i, j]
            denom = kii + kjj - 2 * kij + 1e-12

            raw_d = (g_j - g_i) / denom

            d_max = min(upper - alpha[i], alpha[j])
            d_min = max(-alpha[i], alpha[j] - upper)

            d = np.clip(raw_d, d_min, d_max)

            alpha[i] += d
            alpha[j] -= d

        sv_mask = alpha > self.tol * upper
        self.support_vectors_ = X[sv_mask]
        self.alpha_sv_ = alpha[sv_mask]

        K_sv = self._kernel_matrix(self.support_vectors_, self.support_vectors_, gamma_val)
        self.rho_ = np.mean(self.alpha_sv_ @ K_sv)

        return self

    def decision_function(self, X):
        K_pred = self._kernel_matrix(X, self.support_vectors_, self.gamma_)
        return K_pred @ self.alpha_sv_ - self.rho_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score_samples(self, X):
        return self.decision_function(X)

    def __repr__(self):
        return f"OneClassSVM(kernel={self.kernel!r}, nu={self.nu}, gamma={self.gamma!r})"
