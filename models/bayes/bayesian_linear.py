import numpy as np


class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0, fit_intercept=True):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def _augment(self, X):
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        X_aug = self._augment(X)
        n_features_aug = X_aug.shape[1]
        self.n_features_in_ = X.shape[1]
        S_N_inv = self.alpha * np.eye(n_features_aug) + self.beta * X_aug.T @ X_aug
        self.posterior_cov_ = np.linalg.inv(S_N_inv)
        self.posterior_mean_ = self.beta * self.posterior_cov_ @ X_aug.T @ y
        return self

    def predict(self, X):
        X_aug = self._augment(X)
        return X_aug @ self.posterior_mean_

    def predict_with_uncertainty(self, X):
        X_aug = self._augment(X)
        mean = X_aug @ self.posterior_mean_
        variance = 1.0 / self.beta + np.sum((X_aug @ self.posterior_cov_) * X_aug, axis=1)
        std = np.sqrt(variance)
        return mean, std

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot

    def __repr__(self):
        return (
            f"BayesianLinearRegression(alpha={self.alpha}, "
            f"beta={self.beta}, fit_intercept={self.fit_intercept})"
        )
