import numpy as np


class GaussianMixture:
    def __init__(self, n_components=1, covariance_type='full', max_iter=100,
                 tol=1e-3, n_init=1, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _log_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean
        try:
            L = np.linalg.cholesky(cov)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff.T).T
            maha = np.sum(solved ** 2, axis=1)
        except np.linalg.LinAlgError:
            cov_reg = cov + 1e-6 * np.eye(n_features)
            log_det = np.log(np.linalg.det(cov_reg) + 1e-300)
            inv_cov = np.linalg.inv(cov_reg)
            maha = np.sum(diff @ inv_cov * diff, axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + maha)

    def _e_step(self, X):
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov_k = self.covariances_[k]
            elif self.covariance_type == 'tied':
                cov_k = self.covariances_
            elif self.covariance_type == 'diag':
                cov_k = np.diag(self.covariances_[k])
            elif self.covariance_type == 'spherical':
                cov_k = self.covariances_[k] * np.eye(X.shape[1])
            else:
                cov_k = self.covariances_[k]
            log_resp[:, k] = np.log(self.weights_[k] + 1e-300) + self._log_pdf(X, self.means_[k], cov_k)

        log_sum = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
        log_resp -= log_sum
        return log_resp, log_sum.squeeze()

    def _m_step(self, X, log_resp):
        n_samples, n_features = X.shape
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 1e-10

        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ X) / nk[:, None]

        if self.covariance_type == 'full':
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k:k+1] * diff).T @ diff / nk[k]
                self.covariances_[k] += 1e-6 * np.eye(n_features)

        elif self.covariance_type == 'tied':
            cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov += (resp[:, k:k+1] * diff).T @ diff
            self.covariances_ = cov / n_samples + 1e-6 * np.eye(n_features)

        elif self.covariance_type == 'diag':
            self.covariances_ = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k:k+1] * diff ** 2, axis=0) / nk[k] + 1e-6

        elif self.covariance_type == 'spherical':
            self.covariances_ = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k] * np.sum(diff ** 2, axis=1)) / (nk[k] * n_features) + 1e-6

    def _init_params(self, X, rng):
        n_samples, n_features = X.shape
        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.weights_ = np.ones(self.n_components) / self.n_components

        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances_ = np.eye(n_features)
        elif self.covariance_type == 'diag':
            self.covariances_ = np.ones((self.n_components, n_features))
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.ones(self.n_components)

    def _compute_log_likelihood(self, log_sum):
        return float(np.mean(log_sum))

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        best_ll = -np.inf
        best_params = None

        for _ in range(self.n_init):
            self._init_params(X, rng)
            prev_ll = -np.inf
            self.converged_ = False

            for iteration in range(self.max_iter):
                log_resp, log_sum = self._e_step(X)
                self._m_step(X, log_resp)
                ll = self._compute_log_likelihood(log_sum)
                if abs(ll - prev_ll) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
                prev_ll = ll
            else:
                self.n_iter_ = self.max_iter

            if prev_ll > best_ll:
                best_ll = prev_ll
                best_params = {
                    'weights_': self.weights_.copy(),
                    'means_': self.means_.copy(),
                    'covariances_': (self.covariances_.copy()
                                     if hasattr(self.covariances_, 'copy')
                                     else self.covariances_),
                    'converged_': self.converged_,
                    'n_iter_': self.n_iter_,
                }

        self.weights_ = best_params['weights_']
        self.means_ = best_params['means_']
        self.covariances_ = best_params['covariances_']
        self.converged_ = best_params['converged_']
        self.n_iter_ = best_params['n_iter_']
        return self

    def predict(self, X):
        log_resp, _ = self._e_step(X)
        return np.argmax(log_resp, axis=1)

    def predict_proba(self, X):
        log_resp, _ = self._e_step(X)
        return np.exp(log_resp)

    def score(self, X):
        _, log_sum = self._e_step(X)
        return float(np.mean(log_sum))

    def sample(self, n_samples):
        rng = np.random.RandomState(self.random_state)
        component_counts = rng.multinomial(n_samples, self.weights_)
        samples = []
        n_features = self.means_.shape[1]
        for k, count in enumerate(component_counts):
            if count == 0:
                continue
            if self.covariance_type == 'full':
                cov_k = self.covariances_[k]
            elif self.covariance_type == 'tied':
                cov_k = self.covariances_
            elif self.covariance_type == 'diag':
                cov_k = np.diag(self.covariances_[k])
            elif self.covariance_type == 'spherical':
                cov_k = self.covariances_[k] * np.eye(n_features)
            else:
                cov_k = self.covariances_[k]
            try:
                L = np.linalg.cholesky(cov_k)
                z = rng.randn(count, n_features)
                samples.append(self.means_[k] + z @ L.T)
            except np.linalg.LinAlgError:
                z = rng.randn(count, n_features)
                samples.append(self.means_[k] + z * np.sqrt(np.diag(cov_k)))
        return np.vstack(samples)

    def __repr__(self):
        return (f"GaussianMixture(n_components={self.n_components}, "
                f"covariance_type={self.covariance_type!r}, "
                f"max_iter={self.max_iter})")
