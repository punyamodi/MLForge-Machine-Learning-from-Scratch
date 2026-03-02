import numpy as np


class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        self.C = C
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
            sq_dist = (np.sum(X**2, axis=1, keepdims=True)
                       + np.sum(Y**2, axis=1)
                       - 2 * X @ Y.T)
            return np.exp(-gamma * np.maximum(sq_dist, 0))
        if self.kernel == 'poly':
            return (gamma * X @ Y.T + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return np.tanh(gamma * X @ Y.T + self.coef0)
        return X @ Y.T

    def _smo(self, K, y, C, tol, max_iter):
        n = len(y)
        alpha = np.zeros(n)
        b = 0.0

        for _ in range(max_iter):
            n_changed = 0
            for i in range(n):
                Ei = float(np.dot(alpha * y, K[i])) + b - y[i]
                if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    Ej = float(np.dot(alpha * y, K[j])) + b - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(C, C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - C)
                        H = min(C, ai_old + aj_old)
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
                    b1 = (b - Ei
                          - y[i] * (alpha[i] - ai_old) * K[i, i]
                          - y[j] * (alpha[j] - aj_old) * K[i, j])
                    b2 = (b - Ej
                          - y[i] * (alpha[i] - ai_old) * K[i, j]
                          - y[j] * (alpha[j] - aj_old) * K[j, j])
                    if 0 < alpha[i] < C:
                        b = b1
                    elif 0 < alpha[j] < C:
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
        self.X_train_ = X.copy()
        self.y_train_ = y_enc.copy()
        K = self._kernel_func(X, X, self.gamma_)
        self.alpha_, self.b_ = self._smo(K, y_enc, self.C, self.tol, self.max_iter)
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

    def predict_proba(self, X):
        df = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-df))
        return np.column_stack([1 - proba_pos, proba_pos])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def __repr__(self):
        return f"SVC(C={self.C}, kernel={self.kernel!r}, gamma={self.gamma!r})"


class SVR:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 epsilon=0.1, tol=1e-3, max_iter=1000, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.epsilon = epsilon
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
            sq_dist = (np.sum(X**2, axis=1, keepdims=True)
                       + np.sum(Y**2, axis=1)
                       - 2 * X @ Y.T)
            return np.exp(-gamma * np.maximum(sq_dist, 0))
        if self.kernel == 'poly':
            return (gamma * X @ Y.T + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return np.tanh(gamma * X @ Y.T + self.coef0)
        return X @ Y.T

    def _smo_svr(self, K, y, C, epsilon, tol, max_iter):
        n = len(y)
        alpha = np.zeros(n)
        alpha_star = np.zeros(n)
        b = 0.0

        for _ in range(max_iter):
            n_changed = 0
            for i in range(n):
                w_dot_xi = float(np.dot((alpha - alpha_star), K[i]))
                fi = w_dot_xi + b
                ri = fi - y[i]

                if ri > epsilon + tol and alpha[i] < C:
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    fj = float(np.dot((alpha - alpha_star), K[j])) + b
                    rj = fj - y[j]

                    ai_old = alpha[i]
                    aj_old = alpha[j]

                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        eta = 1e-12

                    delta = -(ri - rj) / eta
                    alpha[i] = np.clip(ai_old + delta, 0, C)
                    alpha[j] = np.clip(aj_old - delta, 0, C)

                    if abs(alpha[i] - ai_old) > 1e-5:
                        b -= ri + (alpha[i] - ai_old) * K[i, i] - (alpha[j] - aj_old) * K[i, j]
                        n_changed += 1

                elif ri < -(epsilon + tol) and alpha_star[i] < C:
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    fj = float(np.dot((alpha - alpha_star), K[j])) + b
                    rj = fj - y[j]

                    asi_old = alpha_star[i]
                    asj_old = alpha_star[j]

                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        eta = 1e-12

                    delta = (ri - rj) / eta
                    alpha_star[i] = np.clip(asi_old + delta, 0, C)
                    alpha_star[j] = np.clip(asj_old - delta, 0, C)

                    if abs(alpha_star[i] - asi_old) > 1e-5:
                        b += abs(ri) - (alpha_star[i] - asi_old) * K[i, i] + (alpha_star[j] - asj_old) * K[i, j]
                        n_changed += 1

            if n_changed == 0:
                break

        return alpha, alpha_star, b

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.gamma_ = self._get_gamma(X)
        K = self._kernel_func(X, X, self.gamma_)
        self.alpha_, self.alpha_star_, self.b_ = self._smo_svr(
            K, y, self.C, self.epsilon, self.tol, self.max_iter
        )
        sv_mask = (self.alpha_ > 1e-5) | (self.alpha_star_ > 1e-5)
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = (self.alpha_ - self.alpha_star_)[sv_mask]
        self.n_support_ = int(sv_mask.sum())
        return self

    def predict(self, X):
        K = self._kernel_func(X, self.support_vectors_, self.gamma_)
        return K @ self.dual_coef_ + self.b_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def __repr__(self):
        return (f"SVR(C={self.C}, kernel={self.kernel!r}, "
                f"epsilon={self.epsilon}, gamma={self.gamma!r})")
