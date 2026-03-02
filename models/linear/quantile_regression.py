import numpy as np


class QuantileRegression:
    def __init__(self, quantile=0.5, learning_rate=0.01, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
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

        n_samples, n_features = X_b.shape
        theta = np.zeros(n_features)

        for iteration in range(self.max_iter):
            residuals = y - X_b @ theta
            grad_sign = self.quantile - (residuals < 0).astype(float)
            gradient = -X_b.T @ grad_sign / n_samples
            theta_new = theta - self.learning_rate * gradient
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
        residuals = y - y_pred
        pinball = np.where(residuals >= 0, self.quantile * residuals, (self.quantile - 1.0) * residuals)
        return -np.mean(pinball)

    def __repr__(self):
        return (
            f"QuantileRegression(quantile={self.quantile}, learning_rate={self.learning_rate}, "
            f"max_iter={self.max_iter}, tol={self.tol}, fit_intercept={self.fit_intercept})"
        )
