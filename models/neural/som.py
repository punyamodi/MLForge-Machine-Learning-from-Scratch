import numpy as np


class SelfOrganizingMap:
    def __init__(self, n_rows=10, n_cols=10, n_iter=1000, learning_rate=0.5,
                 sigma=1.0, random_state=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.weights_ = rng.randn(self.n_rows, self.n_cols, n_features) * 0.1
        grid_rows, grid_cols = np.meshgrid(np.arange(self.n_rows), np.arange(self.n_cols), indexing='ij')
        self.grid_positions_ = np.stack([grid_rows, grid_cols], axis=-1).reshape(-1, 2)

        for t in range(self.n_iter):
            lr = self.learning_rate * np.exp(-t / self.n_iter)
            sigma_t = self.sigma * np.exp(-t / self.n_iter)
            idx = rng.randint(0, X.shape[0])
            x = X[idx]
            dists = np.sum((self.weights_ - x) ** 2, axis=-1)
            bmu_row, bmu_col = np.unravel_index(np.argmin(dists), (self.n_rows, self.n_cols))
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    d = (r - bmu_row) ** 2 + (c - bmu_col) ** 2
                    h = np.exp(-d / (2 * sigma_t ** 2 + 1e-10))
                    self.weights_[r, c] += lr * h * (x - self.weights_[r, c])

        self.labels_ = self.transform(X)
        return self

    def winner(self, x):
        dists = np.sum((self.weights_ - x) ** 2, axis=-1)
        bmu_row, bmu_col = np.unravel_index(np.argmin(dists), (self.n_rows, self.n_cols))
        return (bmu_row, bmu_col)

    def transform(self, X):
        coords = []
        for x in X:
            r, c = self.winner(x)
            coords.append([r, c])
        return np.array(coords)

    def quantization_error(self, X):
        errors = []
        for x in X:
            r, c = self.winner(x)
            errors.append(np.linalg.norm(x - self.weights_[r, c]))
        return np.mean(errors)

    def __repr__(self):
        return f"SelfOrganizingMap(n_rows={self.n_rows}, n_cols={self.n_cols}, n_iter={self.n_iter})"
