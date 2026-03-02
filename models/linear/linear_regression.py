import numpy as np


class LinearRegression:
    def __init__(self, method='normal', learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
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

        if self.method == 'normal':
            result, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            if self.fit_intercept:
                self.intercept_ = result[0]
                self.coef_ = result[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = result

        elif self.method == 'gd':
            n_samples, n_features = X_b.shape
            theta = np.zeros(n_features)
            for _ in range(self.n_iterations):
                residuals = X_b @ theta - y
                gradient = X_b.T @ residuals / n_samples
                theta -= self.learning_rate * gradient
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
            f"LinearRegression(method={self.method!r}, "
            f"learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"fit_intercept={self.fit_intercept})"
        )
