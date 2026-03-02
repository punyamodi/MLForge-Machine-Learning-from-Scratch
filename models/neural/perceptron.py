import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000, eta0=0.01, tol=1e-3, fit_intercept=True, random_state=None):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = np.zeros(1)
        y_enc = np.where(y == self.classes_[0], -1, 1)
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y_enc):
                pred = 1 if (np.dot(xi, self.coef_) + self.intercept_[0]) >= 0 else -1
                if pred != yi:
                    self.coef_ += self.eta0 * yi * xi
                    self.intercept_ += self.eta0 * yi
                    errors += 1
            self.n_iter_ += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        scores = X @ self.coef_ + self.intercept_[0]
        enc = np.where(scores >= 0, 1, -1)
        return np.where(enc == 1, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"Perceptron(max_iter={self.max_iter}, eta0={self.eta0})"
