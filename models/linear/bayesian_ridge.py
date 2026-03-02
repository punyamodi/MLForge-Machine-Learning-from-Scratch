import numpy as np


class BayesianRidge:
    def __init__(self, n_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, fit_intercept=True):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.alpha_ = None
        self.lambda_ = None
        self.sigma_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c = X
            y_c = y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n_samples, n_features = X_c.shape

        self.alpha_ = 1.0
        self.lambda_ = 1.0

        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        S2 = S ** 2

        for iteration in range(self.n_iter):
            alpha_old = self.alpha_
            lambda_old = self.lambda_

            denom_diag = self.lambda_ + self.alpha_ * S2
            m_N = Vt.T @ (self.alpha_ * S / denom_diag * (U.T @ y_c))

            gamma = np.sum(self.alpha_ * S2 / denom_diag)

            denom_lambda = np.dot(m_N, m_N) + 2.0 * self.lambda_2
            if denom_lambda > 1e-10:
                self.lambda_ = (2.0 * self.lambda_1 + gamma) / denom_lambda
            residuals = y_c - X_c @ m_N
            denom_alpha = np.dot(residuals, residuals) + 2.0 * self.alpha_2
            if denom_alpha > 1e-10:
                self.alpha_ = (2.0 * self.alpha_1 + n_samples - gamma) / denom_alpha

            self.alpha_ = max(self.alpha_, 1e-10)
            self.lambda_ = max(self.lambda_, 1e-10)

            delta_alpha = np.abs(self.alpha_ - alpha_old)
            delta_lambda = np.abs(self.lambda_ - lambda_old)
            if delta_alpha < self.tol and delta_lambda < self.tol:
                break

        S_diag = 1.0 / (self.lambda_ + self.alpha_ * S2)
        self.sigma_ = Vt.T @ (S_diag[:, np.newaxis] * Vt)
        self.coef_ = m_N

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def predict_std(self, X):
        X = np.array(X, dtype=float)
        if self.fit_intercept:
            X_c = X - X.mean(axis=0)
        else:
            X_c = X
        var = 1.0 / self.alpha_ + np.sum(X_c @ self.sigma_ * X_c, axis=1)
        return np.sqrt(np.maximum(var, 0.0))

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (
            f"BayesianRidge(n_iter={self.n_iter}, tol={self.tol}, "
            f"alpha_1={self.alpha_1}, alpha_2={self.alpha_2}, "
            f"lambda_1={self.lambda_1}, lambda_2={self.lambda_2}, "
            f"fit_intercept={self.fit_intercept})"
        )
