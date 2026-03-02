import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c = X
            y_c = y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n_features = X_c.shape[1]
        A = X_c.T @ X_c + self.alpha * np.eye(n_features)
        b = X_c.T @ y_c
        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

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
        return f"RidgeRegression(alpha={self.alpha}, fit_intercept={self.fit_intercept})"
