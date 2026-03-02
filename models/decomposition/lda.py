import numpy as np
import scipy.linalg


class LinearDiscriminantAnalysis:
    def __init__(self, n_components=None, tol=1e-4):
        self.n_components = n_components
        self.tol = tol

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        overall_mean = X.mean(axis=0)
        self.means_ = {}
        for cls in self.classes_:
            self.means_[cls] = X[y == cls].mean(axis=0)

        Sw = np.zeros((n_features, n_features))
        for cls in self.classes_:
            X_cls = X[y == cls]
            diff = X_cls - self.means_[cls]
            Sw += diff.T @ diff

        Sb = np.zeros((n_features, n_features))
        for cls in self.classes_:
            n_cls = np.sum(y == cls)
            diff = (self.means_[cls] - overall_mean).reshape(-1, 1)
            Sb += n_cls * (diff @ diff.T)

        reg = self.tol * np.eye(n_features)
        eigenvalues, eigenvectors = scipy.linalg.eigh(Sb, Sw + reg)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        n_comp = self.n_components if self.n_components is not None else min(n_classes - 1, n_features)
        n_comp = min(n_comp, n_features)
        self.scalings_ = eigenvectors[:, :n_comp]
        self.n_components_ = n_comp

        means_proj = np.array([self.means_[cls] @ self.scalings_ for cls in self.classes_])
        self.coef_ = self.scalings_
        self.intercept_ = -0.5 * np.sum(means_proj ** 2, axis=1)

        return self

    def transform(self, X):
        return X @ self.scalings_

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        X_proj = self.transform(X)
        means_proj = np.array([self.means_[cls] @ self.scalings_ for cls in self.classes_])
        dists = np.sum((X_proj[:, np.newaxis, :] - means_proj[np.newaxis, :, :]) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"LinearDiscriminantAnalysis(n_components={self.n_components}, tol={self.tol})"
