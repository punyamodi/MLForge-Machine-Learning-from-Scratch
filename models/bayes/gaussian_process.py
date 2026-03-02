import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


class RBFKernel:
    def __init__(self, length_scale=1.0, variance=1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X, Y):
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_dists = X_sq + Y_sq.T - 2.0 * X @ Y.T
        sq_dists = np.maximum(sq_dists, 0.0)
        return self.variance * np.exp(-0.5 * sq_dists / self.length_scale ** 2)

    def diag(self, X):
        return self.variance * np.ones(X.shape[0])

    def __repr__(self):
        return f"RBFKernel(length_scale={self.length_scale}, variance={self.variance})"


class MaternKernel:
    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, X, Y):
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_dists = X_sq + Y_sq.T - 2.0 * X @ Y.T
        sq_dists = np.maximum(sq_dists, 0.0)
        r = np.sqrt(sq_dists)
        l = self.length_scale
        if self.nu == 0.5:
            return np.exp(-r / l)
        elif self.nu == 1.5:
            sqrt3_r_l = np.sqrt(3.0) * r / l
            return (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif self.nu == 2.5:
            sqrt5_r_l = np.sqrt(5.0) * r / l
            return (1.0 + sqrt5_r_l + 5.0 * r ** 2 / (3.0 * l ** 2)) * np.exp(-sqrt5_r_l)
        else:
            raise ValueError(f"Unsupported nu={self.nu}. Choose from {{0.5, 1.5, 2.5}}")

    def diag(self, X):
        return np.ones(X.shape[0])

    def __repr__(self):
        return f"MaternKernel(length_scale={self.length_scale}, nu={self.nu})"


class LinearKernel:
    def __init__(self, sigma_b=1.0, sigma_v=1.0):
        self.sigma_b = sigma_b
        self.sigma_v = sigma_v

    def __call__(self, X, Y):
        return self.sigma_b ** 2 + self.sigma_v ** 2 * X @ Y.T

    def diag(self, X):
        return self.sigma_b ** 2 + self.sigma_v ** 2 * np.sum(X ** 2, axis=1)

    def __repr__(self):
        return f"LinearKernel(sigma_b={self.sigma_b}, sigma_v={self.sigma_v})"


class GaussianProcessRegressor:
    def __init__(self, kernel=None, alpha=1e-10, normalize_y=False):
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.alpha = alpha
        self.normalize_y = normalize_y

    def fit(self, X, y):
        self.X_train_ = X.copy()
        if self.normalize_y:
            self.y_mean_ = y.mean()
            self.y_std_ = y.std()
            if self.y_std_ == 0.0:
                self.y_std_ = 1.0
            y_normalized = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0.0
            self.y_std_ = 1.0
            y_normalized = y
        K = self.kernel(X, X) + self.alpha * np.eye(X.shape[0])
        self.L_, self.low_ = cho_factor(K, lower=True)
        self.alpha_vec_ = cho_solve((self.L_, self.low_), y_normalized)
        return self

    def predict(self, X, return_std=False):
        k_star = self.kernel(X, self.X_train_)
        mean = k_star @ self.alpha_vec_
        if self.normalize_y:
            mean = mean * self.y_std_ + self.y_mean_
        if return_std:
            v = solve_triangular(self.L_, k_star.T, lower=True)
            var = self.kernel.diag(X) - np.sum(v ** 2, axis=0)
            std = np.sqrt(np.clip(var, 0, None))
            return mean, std
        return mean

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot

    def __repr__(self):
        return (
            f"GaussianProcessRegressor(kernel={self.kernel}, "
            f"alpha={self.alpha}, normalize_y={self.normalize_y})"
        )


class GaussianProcessClassifier:
    def __init__(self, kernel=None, max_iter=100, tol=1e-5):
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.max_iter = max_iter
        self.tol = tol

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_encoded = (y == self.classes_[1]).astype(float)
        self.X_train_ = X.copy()
        n = X.shape[0]
        K = self.kernel(X, X)
        f = np.zeros(n)
        for _ in range(self.max_iter):
            pi = self._sigmoid(f)
            W_diag = pi * (1.0 - pi)
            sqrt_W = np.sqrt(W_diag)
            B = np.eye(n) + (sqrt_W[:, None] * K) * sqrt_W[None, :]
            L_factor = cho_factor(B, lower=True)
            b = W_diag * f + (y_encoded - pi)
            inner = cho_solve(L_factor, sqrt_W * (K @ b))
            a = b - sqrt_W * inner
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new
        pi = self._sigmoid(f)
        W_diag = pi * (1.0 - pi)
        sqrt_W = np.sqrt(W_diag)
        B = np.eye(n) + (sqrt_W[:, None] * K) * sqrt_W[None, :]
        self.L_factor_ = cho_factor(B, lower=True)
        self.W_hat_ = W_diag
        self.f_hat_ = f
        self.a_ = a
        return self

    def predict_proba(self, X):
        k_star = self.kernel(X, self.X_train_)
        f_bar = k_star @ self.a_
        sqrt_W_hat = np.sqrt(self.W_hat_)
        v = solve_triangular(
            self.L_factor_[0],
            (sqrt_W_hat[:, None] * k_star.T),
            lower=True
        )
        var = self.kernel.diag(X) - np.sum(v ** 2, axis=0)
        kappa = 1.0 / np.sqrt(1.0 + np.pi / 8.0 * var)
        p = self._sigmoid(kappa * f_bar)
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (
            f"GaussianProcessClassifier(kernel={self.kernel}, "
            f"max_iter={self.max_iter}, tol={self.tol})"
        )
