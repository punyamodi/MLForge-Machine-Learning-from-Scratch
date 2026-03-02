import numpy as np


class PCA:
    def __init__(self, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = self.n_components if self.n_components is not None else X.shape[1]
        n_components = min(n_components, min(X.shape))
        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]
        self.explained_variance_ = (S[:n_components] ** 2) / (X.shape[0] - 1)
        total_var = (S ** 2).sum() / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.n_components_ = n_components
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T
        if self.whiten:
            X_transformed /= (self.singular_values_[:self.n_components_] + 1e-10)
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        if self.whiten:
            X_transformed = X_transformed * self.singular_values_[:self.n_components_]
        return X_transformed @ self.components_ + self.mean_

    def __repr__(self):
        return f"PCA(n_components={self.n_components}, whiten={self.whiten})"
