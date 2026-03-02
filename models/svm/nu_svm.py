import numpy as np


class NuSVC:
    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * X.var())
        if self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _kernel_func(self, X, Y, gamma):
        if self.kernel == 'linear':
            return X @ Y.T
        if self.kernel == 'rbf':
            sq_dist = (np.sum(X ** 2, axis=1, keepdims=True)
                       + np.sum(Y ** 2, axis=1)
                       - 2 * X @ Y.T)
            return np.exp(-gamma * np.maximum(sq_dist, 0))
        if self.kernel == 'poly':
            return (gamma * X @ Y.T + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return np.tanh(gamma * X @ Y.T + self.coef0)
        return X @ Y.T

    def _nu_smo(self, K, y, nu, tol, max_iter):
        n = len(y)
        upper_bound = 1.0 / n
        alpha = np.full(n, nu / 2.0)
        b = 0.0
        C_eff = upper_bound

        for _ in range(max_iter):
            n_changed = 0
            for i in range(n):
                Ei = float(np.dot(alpha * y, K[i])) + b - y[i]
                if (y[i] * Ei < -tol and alpha[i] < C_eff) or (y[i] * Ei > tol and alpha[i] > 0):
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    Ej = float(np.dot(alpha * y, K[j])) + b - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(C_eff, C_eff + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - C_eff)
                        H = min(C_eff, ai_old + aj_old)

                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue

                    alpha[i] += y[i] * y[j] * (aj_old - alpha[j])

                    alpha[i] = np.clip(alpha[i], 0.0, C_eff)
                    alpha[j] = np.clip(alpha[j], 0.0, C_eff)

                    sum_alpha_y = float(np.dot(alpha, y))
                    if abs(sum_alpha_y) > tol:
                        correction = sum_alpha_y / n
                        alpha -= correction * y
                        alpha = np.clip(alpha, 0.0, C_eff)

                    b1 = (b - Ei
                          - y[i] * (alpha[i] - ai_old) * K[i, i]
                          - y[j] * (alpha[j] - aj_old) * K[i, j])
                    b2 = (b - Ej
                          - y[i] * (alpha[i] - ai_old) * K[i, j]
                          - y[j] * (alpha[j] - aj_old) * K[j, j])

                    if 0 < alpha[i] < C_eff:
                        b = b1
                    elif 0 < alpha[j] < C_eff:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    n_changed += 1

            if n_changed == 0:
                break

        return alpha, b

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        y_enc = np.where(y == self.classes_[0], -1.0, 1.0)
        self.gamma_ = self._get_gamma(X)
        K = self._kernel_func(X, X, self.gamma_)
        self.alpha_, self.b_ = self._nu_smo(K, y_enc, self.nu, self.tol, self.max_iter)
        sv_mask = self.alpha_ > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_alpha_ = self.alpha_[sv_mask]
        self.support_y_ = y_enc[sv_mask]
        self.n_support_ = int(sv_mask.sum())
        return self

    def decision_function(self, X):
        K = self._kernel_func(X, self.support_vectors_, self.gamma_)
        return K @ (self.support_alpha_ * self.support_y_) + self.b_

    def predict(self, X):
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def __repr__(self):
        return f"NuSVC(nu={self.nu}, kernel={self.kernel!r}, gamma={self.gamma!r})"
