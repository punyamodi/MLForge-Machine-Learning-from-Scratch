import numpy as np
from numpy.linalg import inv, eigh


class EllipticEnvelope:
    def __init__(self, contamination=0.1, support_fraction=None, random_state=None):
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state

    def _mahalanobis_distances(self, X, loc, prec):
        diff = X - loc
        left = diff @ prec
        maha_sq = np.sum(left * diff, axis=1)
        maha_sq = np.maximum(maha_sq, 0.0)
        return np.sqrt(maha_sq)

    def _mcd_c_step(self, X, h, mean_init, cov_init, n_iter=30):
        n, p = X.shape
        mean_cur = mean_init.copy()
        cov_cur = cov_init.copy()

        for _ in range(n_iter):
            reg_cov = cov_cur + 1e-6 * np.eye(p)
            try:
                prec = inv(reg_cov)
            except np.linalg.LinAlgError:
                prec = np.eye(p)

            dists = self._mahalanobis_distances(X, mean_cur, prec)
            h_idx = np.argsort(dists)[:h]
            X_h = X[h_idx]

            mean_new = X_h.mean(axis=0)
            diff = X_h - mean_new
            cov_new = (diff.T @ diff) / h

            mean_diff = np.max(np.abs(mean_new - mean_cur))
            cov_diff = np.max(np.abs(cov_new - cov_cur))

            mean_cur = mean_new
            cov_cur = cov_new

            if mean_diff < 1e-8 and cov_diff < 1e-8:
                break

        return mean_cur, cov_cur

    def fit(self, X):
        n, p = X.shape
        if self.support_fraction is not None:
            h = int(self.support_fraction * n)
        else:
            h = int((n + p + 1) / 2)
        h = max(h, p + 1)
        h = min(h, n)

        rng = np.random.RandomState(self.random_state if self.random_state is not None else 0)

        best_mean = None
        best_cov = None
        best_det = np.inf

        for _ in range(5):
            idx = rng.choice(n, size=p + 1, replace=False)
            X_init = X[idx]
            mean_init = X_init.mean(axis=0)
            diff = X_init - mean_init
            cov_init = (diff.T @ diff) / (p + 1)
            cov_init += 1e-6 * np.eye(p)

            mean_trial, cov_trial = self._mcd_c_step(X, h, mean_init, cov_init)

            reg_cov_trial = cov_trial + 1e-6 * np.eye(p)
            try:
                det = np.linalg.det(reg_cov_trial)
            except Exception:
                det = np.inf

            if det < best_det:
                best_det = det
                best_mean = mean_trial
                best_cov = reg_cov_trial

        self.location_ = best_mean
        self.covariance_ = best_cov + 1e-6 * np.eye(p)
        self.precision_ = inv(self.covariance_)

        dist = self._mahalanobis_distances(X, self.location_, self.precision_)
        self.threshold_ = np.percentile(dist ** 2, 100 * (1 - self.contamination))

        return self

    def mahalanobis(self, X):
        return self._mahalanobis_distances(X, self.location_, self.precision_)

    def decision_function(self, X):
        return self.threshold_ - self.mahalanobis(X) ** 2

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def __repr__(self):
        return f"EllipticEnvelope(contamination={self.contamination})"
