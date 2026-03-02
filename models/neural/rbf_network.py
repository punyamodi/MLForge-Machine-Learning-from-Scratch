import numpy as np


class RBFNetwork:
    def __init__(self, n_centers=10, gamma=1.0, random_state=None):
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state

    def _kmeans(self, X, k, n_iter=300):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(X.shape[0], k, replace=False)
        centers = X[idx].copy()
        for _ in range(n_iter):
            dists = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                for j in range(k)
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers
        return centers

    def _rbf_features(self, X):
        dists = np.sum((X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * dists)

    def fit(self, X, y):
        y_arr = np.array(y)
        is_classification = not np.issubdtype(y_arr.dtype, np.floating)

        self.centers_ = self._kmeans(X, self.n_centers)
        Phi = self._rbf_features(X)
        Phi_bias = np.hstack([Phi, np.ones((Phi.shape[0], 1))])

        if is_classification:
            self._is_classification = True
            self.classes_ = np.unique(y_arr)
            n_classes = len(self.classes_)
            if n_classes == 2:
                y_enc = (y_arr == self.classes_[1]).astype(float).reshape(-1, 1)
            else:
                y_enc = np.zeros((len(y_arr), n_classes))
                for i, cls in enumerate(self.classes_):
                    y_enc[y_arr == cls, i] = 1.0
            sol, _, _, _ = np.linalg.lstsq(Phi_bias, y_enc, rcond=None)
            self.weights_ = sol[:-1]
            self.bias_ = sol[-1]
        else:
            self._is_classification = False
            y_enc = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            sol, _, _, _ = np.linalg.lstsq(Phi_bias, y_enc, rcond=None)
            self.weights_ = sol[:-1]
            self.bias_ = sol[-1]

        return self

    def predict(self, X):
        Phi = self._rbf_features(X)
        out = Phi @ self.weights_ + self.bias_

        if self._is_classification:
            idx = np.argmax(out, axis=1)
            return self.classes_[idx]
        else:
            return out.ravel() if out.shape[1] == 1 else out

    def __repr__(self):
        return f"RBFNetwork(n_centers={self.n_centers}, gamma={self.gamma})"
