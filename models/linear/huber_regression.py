import numpy as np


class HuberRegression:
    def __init__(self, epsilon=1.35, max_iter=100, tol=1e-4, alpha=0.0001, fit_intercept=True):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
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
        theta = np.zeros(n_features)

        for iteration in range(self.max_iter):
            residuals = y - X_b @ theta
            abs_res = np.abs(residuals)
            weights = np.where(abs_res <= self.epsilon, 1.0, self.epsilon / (abs_res + 1e-10))
            W = np.diag(weights)
            A = X_b.T @ W @ X_b + self.alpha * np.eye(n_features)
            b = X_b.T @ W @ y
            theta_new = np.linalg.solve(A, b)
            if np.linalg.norm(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (
            f"HuberRegression(epsilon={self.epsilon}, max_iter={self.max_iter}, "
            f"tol={self.tol}, alpha={self.alpha}, fit_intercept={self.fit_intercept})"
        )
