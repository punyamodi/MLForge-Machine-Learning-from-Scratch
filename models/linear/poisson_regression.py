import numpy as np


class PoissonRegression:
    def __init__(self, max_iter=100, tol=1e-4, fit_intercept=True, alpha=0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if self.fit_intercept:
            X_b = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_b = X

        n_features = X_b.shape[1]
        w = np.zeros(n_features)

        for iteration in range(self.max_iter):
            eta = X_b @ w
            mu = np.exp(np.clip(eta, -500, 500))
            mu = np.maximum(mu, 1e-10)
            W_diag = mu
            z = eta + (y - mu) / mu
            W = np.diag(W_diag)
            A = X_b.T @ W @ X_b + self.alpha * np.eye(n_features)
            b = X_b.T @ (W_diag * z)
            w_new = np.linalg.solve(A, b)
            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break
            w = w_new

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        eta = X @ self.coef_ + self.intercept_
        return np.exp(np.clip(eta, -500, 500))

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        mu = self.predict(X)
        mu = np.maximum(mu, 1e-10)
        y_safe = np.where(y > 0, y, 1e-10)
        deviance = 2.0 * np.sum(y_safe * np.log(y_safe / mu) - (y_safe - mu))
        mu_null = np.mean(y)
        mu_null = max(mu_null, 1e-10)
        null_deviance = 2.0 * np.sum(y_safe * np.log(y_safe / mu_null) - (y_safe - mu_null))
        return 1.0 - deviance / (null_deviance + 1e-10)

    def __repr__(self):
        return (
            f"PoissonRegression(max_iter={self.max_iter}, tol={self.tol}, "
            f"fit_intercept={self.fit_intercept}, alpha={self.alpha})"
        )
