import numpy as np
import itertools


class PolynomialRegression:
    def __init__(self, degree=2, fit_intercept=True):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self._n_input_features = None

    def _polynomial_features(self, X):
        n_samples, n_features = X.shape
        self._n_input_features = n_features
        feature_list = []
        for d in range(1, self.degree + 1):
            for combo in itertools.combinations_with_replacement(range(n_features), d):
                feature_list.append(np.prod(X[:, combo], axis=1))
        return np.column_stack(feature_list)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        X_poly = self._polynomial_features(X)

        if self.fit_intercept:
            X_b = np.column_stack([np.ones(X_poly.shape[0]), X_poly])
        else:
            X_b = X_poly

        result, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = result[0]
            self.coef_ = result[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = result

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        X_poly = self._polynomial_features(X)
        return X_poly @ self.coef_ + self.intercept_

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return f"PolynomialRegression(degree={self.degree}, fit_intercept={self.fit_intercept})"
