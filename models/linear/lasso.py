import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, rho, alpha):
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c = X
            y_c = y
            y_mean = 0.0
            X_mean = np.zeros(n_features)

        self.coef_ = np.zeros(n_features)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                residual = y_c - X_c @ self.coef_ + X_c[:, j] * self.coef_[j]
                rho = X_c[:, j] @ residual
                z_j = np.sum(X_c[:, j] ** 2)
                if z_j == 0:
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(rho / z_j, self.alpha) 
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

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
        return (
            f"LassoRegression(alpha={self.alpha}, max_iter={self.max_iter}, "
            f"tol={self.tol}, fit_intercept={self.fit_intercept})"
        )
